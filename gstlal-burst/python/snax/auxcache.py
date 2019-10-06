# Copyright (C) 2013  Kipp Cannon, Chad Hanna, Drew Keppel
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

## @package auxcache

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import os
import sys
import tempfile


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


from gstlal import datasource
from gstlal import pipeparts
from gstlal import simplehandler
from gstlal.snax import multichannel_datasource

#
# =============================================================================
#
#                           File Clean-Up Machinery
#
# =============================================================================
#


class tempcache(list):
	"""
	List-like object to hold CacheEntry objects, and run os.unlink() on
	the .path of each as they are removed from the list or when the
	list is garbage collected.  All errors during file removal are
	ignored.

	Note that there is no way to remove a CacheEntry from this list
	without the file it represents being deleted.  If, after adding a
	CacheEntry to this list it is decided the file must not be deleted,
	then instead of removing it from the list it must be replaced with
	something else, e.g. None, and that item can then be removed from
	the list.

	Example:

	>>> from lal.utils import CacheEntry
	>>> # create a cache, and add an entry
	>>> cache = tempcache()
	>>> cache.append(CacheEntry("- - - - file://localhost/tmp/blah.txt"))
	>>> # now remove it without the file being deleted
	>>> cache[-1] = None
	>>> del cache[-1]
	"""
	def __delitem__(self, i):
		try:
			os.unlink(self[i].path)
		except:
			pass
		super(tempcache, self).__delitem__(i)

	def __delslice__(self, i, j):
		try:
			for entry in self[i:j]:
				try:
					os.unlink(entry.path)
				except:
					pass
		except:
			pass
		super(tempcache, self).__delslice__(i, j)

	def __del__(self):
		del self[:]


#
# =============================================================================
#
#                                   Handler
#
# =============================================================================
#


class Handler(simplehandler.Handler):
	def __init__(self, *args, **kwargs):
		super(Handler, self).__init__(*args, **kwargs)
		self.cache = tempcache()

	def do_on_message(self, bus, message):
		if message.type == Gst.MessageType.ELEMENT and message.get_structure().get_name() == "GstMultiFileSink":
			self.cache.append(pipeparts.framecpp_filesink_cache_entry_from_mfs_message(message))
			return True
		return False


#
# =============================================================================
#
#                                   Pipeline
#
# =============================================================================
#


def build_pipeline(pipeline, data_source_info, output_path = tempfile.gettempdir(), description = "TMPFILE_DELETE_ME", channel_comment = None, frame_duration = 1, frames_per_file = 64, verbose = False):

	channels = data_source_info.channel_dict.keys()

	#
	# retrieve auxiliary channels
	#

	# FIXME: turning off verbosity since this causes naming conflict with multiple progressreport elements, since they don't have unique identifiers.
	#        should really be using URI handling for this, but that's a separate issue altogether
	src = multichannel_datasource.mkbasicmultisrc(pipeline, data_source_info, channels, verbose = False)

	#
	# pack into frame files for output
	#

	src = pipeparts.mkframecppchannelmux(pipeline, {channel: src[channel] for channel in channels}, frame_duration = frame_duration, frames_per_file = frames_per_file)
	for pad in src.sinkpads:
		if channel_comment is not None:
			pad.set_property("comment", channel_comment)
		pad.set_property("pad-type", "FrProcData")
	pipeparts.mkframecppfilesink(pipeline, src, frame_type = "%s_%s" % (data_source_info.instrument, description), path = output_path)


#
# =============================================================================
#
#                     Collect and Cache Auxiliary Channels
#
# =============================================================================
#


def cache_aux(data_source_info, logger, channel_comment = "cached aux channels", verbose = False, **kwargs):

	#
	# build pipeline
	#

	mainloop = GObject.MainLoop()
	pipeline = Gst.Pipeline(name="pipeline")
	handler = Handler(mainloop, pipeline)


	logger.info("assembling auxiliary channel cache pipeline ...")
	build_pipeline(pipeline, data_source_info, channel_comment = channel_comment, verbose = verbose, **kwargs)

	#
	# seek and run pipeline
	#

	if pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
		raise RuntimeError("pipeline did not enter ready state")
	datasource.pipeline_seek_for_gps(pipeline, *data_source_info.seg)

	logger.info("setting auxiliary channel cache pipeline state to playing ...")
	if pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS:
		raise RuntimeError("pipeline did not enter playing state")

	logger.info("running auxiliary channel cache pipeline ...")
	mainloop.run()
	logger.info("finished producing cached auxiliary frame files.")


	#
	# return tempcache object.  when this object is garbage collected
	# the frame files will be deleted.  keep a reference alive as long
	# as you wish to preserve the files.
	#

	return handler.cache
