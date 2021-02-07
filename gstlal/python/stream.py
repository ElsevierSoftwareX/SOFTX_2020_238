# Copyright (C) 2020  Patrick Godwin
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

## @file

## @package stream

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from collections import namedtuple
from collections.abc import Mapping
import functools
import io
import os
import sys
import uuid

import numpy
import pluggy

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstAudio', '1.0')
from gi.repository import GObject
from gi.repository import Gst
from gi.repository import GstAudio

from ligo import segments

from gstlal import datasource
from gstlal import pipeparts
from gstlal import plugins
from gstlal import simplehandler


#
# =============================================================================
#
#                                    Stream
#
# =============================================================================
#


SourceElem = namedtuple("SourceElem", "datasource is_live gps_range")
Buffer = namedtuple("Buffer", "t0 data")


MessageType = Gst.MessageType


class Stream(object):
	"""Class for building a GStreamer-based pipeline.
	"""
	thread_init = False

	def __init__(self, name=None, mainloop=None, pipeline=None, handler=None, source=None, head=None):
		# initialize threads if not set
		if not self.thread_init:
			GObject.threads_init()
			Gst.init(None)
			self.thread_init = True

		# set up gstreamer pipeline
		self.name = name if name else str(uuid.uuid1())
		self.mainloop = mainloop if mainloop else GObject.MainLoop()
		self.pipeline = pipeline if pipeline else Gst.Pipeline(self.name)
		self.handler = handler if handler else StreamHandler(self.mainloop, self.pipeline)
		self.head = head if head else None

		# set up source elem properties
		self.source = source if source else None

	def start(self):
		"""Start up the pipeline.
		"""
		if self.source.is_live:
			simplehandler.OneTimeSignalHandler(self.pipeline)
		self._set_state(Gst.State.READY)
		if not self.source.is_live:
			self._seek_gps()
		self._set_state(Gst.State.PLAYING)

		## Debugging output
		if os.environ.get("GST_DEBUG_DUMP_DOT_DIR", False):
			name = self.pipeline.get_name()
			pipeparts.write_dump_dot(self.pipeline, f"{name}_PLAYING", verbose=True)

			## Setup a signal handler to intercept SIGINT in order to write
			## the pipeline graph at ctrl+C before cleanly shutting down
			class SigHandler(simplehandler.OneTimeSignalHandler):
				def do_on_call(self, signum, frame):
					pipeparts.write_dump_dot(self.pipeline, f"{name}_SIGINT", verbose=True)
			sighandler = SigHandler(self.pipeline)

		self.mainloop.run()

	@classmethod
	def register_element(cls, elem_name):
		"""Register an element to the stream, making it callable.
		"""
		def register(func):
			def wrapped(self, *srcs, **kwargs):
				head = func(self.pipeline, self.head, *srcs, **kwargs)
				if isinstance(head, Mapping):
					new_head = head.__class__()
					for key, elem in head.items():
						new_head = {
							key: cls(
								name=self.name,
								mainloop=self.mainloop,
								pipeline=self.pipeline,
								handler=self.handler,
								source=self.source,
								head=elem,
							)
						}
					return new_head
				else:
					return cls(
						name=self.name,
						mainloop=self.mainloop,
						pipeline=self.pipeline,
						handler=self.handler,
						source=self.source,
						head=head,
					)
			setattr(cls, elem_name, wrapped)
		return register

	@classmethod
	def from_datasource(cls, data_source_info, ifo, verbose=False):
		stream = cls()
		stream.head, _, _ = datasource.mkbasicsrc(stream.pipeline, data_source_info, ifo, verbose=verbose)
		is_live = data_source_info.data_source in data_source_info.live_sources
		stream.source = SourceElem(
			datasource=data_source_info.data_source,
			is_live=is_live,
			gps_range=data_source_info.seg,
		)
		return stream

	def connect(self, *args, **kwargs):
		self.head.connect(*args, **kwargs)

	def sink(self, func):
		def sample_handler(elem):
			buf = self._pull_buffer(elem)
			func(buf)
			return Gst.FlowReturn.OK

		sink = pipeparts.mkappsink(self.pipeline, self.head, max_buffers=1, sync=False)
		sink.connect("new-sample", sample_handler)
		sink.connect("new-preroll", self._preroll_handler)

	def add_callback(self, msg_type, msg_name, callback):
		"""
		"""
		self.handler.add_callback(msg_type, msg_name, callback)

	def _set_state(self, state):
		"""Set pipeline state, checking for errors.
		"""
		if self.pipeline.set_state(state) == Gst.StateChangeReturn.FAILURE:
			raise RuntimeError(f"pipeline failed to enter {state.value_name}")

	def _seek_gps(self):
		"""Seek pipeline to the given GPS start/end times.
		"""
		start, end = self.source.gps_range
		datasource.pipeline_seek_for_gps(self.pipeline, start, end)

	@staticmethod
	def _pull_buffer(elem):
		buf = elem.emit("pull-sample").get_buffer()
		buftime = buf.pts // 1e9
		result, mapinfo = buf.map(Gst.MapFlags.READ)
		if mapinfo.data:
			with io.BytesIO(mapinfo.data) as s:
				newbuf = Buffer(t0=buftime, data=numpy.loadtxt(s))
		else:
			newbuf = Buffer(t0=buftime, data=None)
		buf.unmap(mapinfo)
		del buf
		return newbuf

	@staticmethod
	def _preroll_handler(elem):
		buf = elem.emit("pull-preroll")
		del buf
		return Gst.FlowReturn.OK


class StreamHandler(simplehandler.Handler):
	def __init__(self, *args, **kwargs): 
		super().__init__(*args, **kwargs)

		# set up callbacks
		self.callbacks = {
			Gst.MessageType.ELEMENT: {},
			Gst.MessageType.APPLICATION: {},
			Gst.MessageType.EOS: {},
		}

	def add_callback(self, msg_type, msg_name, callback):
		"""
		"""
		if msg_name in self.callbacks[msg_type]:
			raise ValueError("callback already registered for message type/name")
		self.callbacks[msg_type][msg_name] = callback

	def do_on_message(self, bus, message):
		"""
		"""
		if message.type in self.callbacks:
			message_name = message.get_structure().get_name()
			if message_name in self.callbacks[message.type]:
				self.callbacks[message.type][message_name](message)
		return False


def _get_registered_elements():
	"""Get all registered GStreamer elements.
	"""
	# set up plugin manager
	manager = pluggy.PluginManager("gstlal")
	manager.add_hookspecs(plugins)
	
	# load elements
	manager.register(pipeparts)

	from gstlal.pipeparts import condition
	manager.register(condition)
	
	# add all registered plugins to registry
	registered = {}
	for plugin_name in manager.hook.elements():
		for name, element in plugin_name.items():
		    registered[name] = element
	
	return registered


# register elements to Stream class
for elem_name, elem in _get_registered_elements().items():
	Stream.register_element(elem_name)(elem)
