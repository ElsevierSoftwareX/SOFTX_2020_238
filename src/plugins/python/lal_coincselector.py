# Copyright (C) 2010 Leo Singer
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
"""
Select interesting coincidences based on estimated false alarm rates.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gstlal.pipeio import net_ifar, sngl_inspiral_groups_from_buffer, sngl_inspiral_groups_to_buffer
try:
	all
except NameError:
	# Python < 2.5 compatibility
	from glue.iterutils import all
import traceback


class TriQueue(object):
	"""A three-element queue, or ring buffer."""

	def __init__(self):
		self.__oldest = None
		self.__middle = None
		self.__newest = None

	def add(self, obj):
		"""Enqueue a new element, discarding the oldest element if necessary."""
		self.__oldest = self.__middle
		self.__middle = self.__newest
		self.__newest = obj

	def is_empty(self):
		"""Determine if the queue is empty."""
		return (self.__oldest is None
			and self.__middle is None
			and self.__newest is None)


	def is_full(self):
		"""Determine if the queue is full."""
		return (self.__oldest is not None
			and self.__middle is not None
			and self.__newest is not None)

	@property
	def top(self):
		"""Find the newest element in the queue."""
		if self.__newest is not None:
			return self.__newest
		elif self.__middle is not None:
			return self.__middle
		else:
			return self.__oldest

	@property
	def oldest(self):
		return self.__oldest

	@property
	def middle(self):
		return self.__middle

	@property
	def newest(self):
		return self.__newest


class CoincBlock(object):
	def __init__(self, timestamp, duration):
		self.timestamp = timestamp
		self.duration = duration
		self.end_time = timestamp + duration
		self.coinc_list = []


class SnglCoinc(object):
	def __init__(self, sngl_group, ifar):
		self.sngl_group = sngl_group
		self.ifar = ifar
		self.time = min(row.end_time * gst.SECOND + row.end_time_ns)


class lal_coincselector(gst.Element):
	__gstdetails__ = (
		'Coincidence Selector',
		'Generic',
		__doc__,
		__author__
	)
	__gproperties__ = {
		'min-ifar': (
			gobject.TYPE_UINT64,
			'Minimum net inverse false alarm rate (IFAR)',
			'Minimum network IFAR (inverse false alarm rate, also known as extrapolated waiting time) in nanoseconds.  Coincidences with an estimated IFAR less than this value will be dropped.',
			0, gst.CLOCK_TIME_NONE, 3600 * gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'min-waiting-time': (
			gobject.TYPE_UINT64,
			'Minimum waiting time',
			'Minimum waiting time between coincidences (nanoseconds).',
			0, gst.CLOCK_TIME_NONE, gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'dt': ( # FIXME: could the coinc and coincselector elements share this piece of information?
			gobject.TYPE_UINT64,
			'Coincidence window',
			'Coincidence window in nanoseconds.',
			0, gst.CLOCK_TIME_NONE, 50 * gst.MSECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
	}
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		),
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		)
	)


	def __init__(self):
		super(lal_coincselector, self).__init__()
		self.create_all_pads()
		self.__srcpad = self.get_static_pad('src')
		self.__sinkpad = self.get_static_pad('sink')
		self.__srcpad.use_fixed_caps() # FIXME: better to use proxycaps?
		self.__sinkpad.use_fixed_caps() # FIXME: better to use proxycaps?
		self.__sinkpad.set_chain_function(self.chain)

		self.__queue = TriQueue()


	def do_set_property(self, prop, val):
		if prop.name == 'min-ifar':
			self.__min_ifar = val
		elif prop.name == 'min-waiting-time':
			self.__min_waiting_time = val
		elif prop.name == 'dt':
			self.__dt = val


	def do_get_property(self, prop):
		if prop.name == 'min-ifar':
			return self.__min_ifar
		elif prop.name == 'min-waiting-time':
			return self.__min_waiting_time
		elif prop.name == 'dt':
			return self.__dt


	def process_coincs(self, pad, inbuf): # FIXME: not a very informative name for this method.
		if self.__queue.is_full():
			rows = []
			coinc_list = self.__queue.middle.coinc_list
			if len(coinc_list) > 0:
				coinc = coinc_list[0]
				for other_coinc in coinc_list[1:]:
					if other_coinc.ifar < coinc.ifar:
						coinc = other_coinc
				if all(x.ifar >= coinc.ifar for x in self.__queue.oldest.coinc_list if coinc.time - x.time < self.__min_waiting_time) and all(x.ifar >= coinc.ifar for x in self.__queue.newest.coinc_list if x.time - coinc.time < self.__min_waiting_time):
					rows = coinc.sngl_group
			outbuf = sngl_inspiral_groups_to_buffer(rows, inbuf.caps[0]['channels'])
			outbuf.timestamp = self.__queue.middle.timestamp
			outbuf.duration = self.__queue.middle.duration
			outbuf.offset = gst.BUFFER_OFFSET_NONE
			outbuf.offset_end = gst.BUFFER_OFFSET_NONE
			outbuf.caps = inbuf.caps
			retval = self.__srcpad.push(outbuf)
		else:
			retval = gst.FLOW_OK
		top = self.__queue.top
		self.__queue.add(CoincBlock(top.timestamp + top.duration, top.duration))
		return retval


	def chain(self, pad, inbuf):
		try: # FIXME: apparently the gst.Pad wrapper silences exceptions from chain() routines.

			# If the queue is completely empty, we need to initialize it by
			# storing the start time of the stream.
			if self.__queue.is_empty():
				top = CoincBlock(inbuf.timestamp, self.__min_waiting_time)
				self.__queue.add(top)
			else:
				top = self.__queue.top

			if inbuf.timestamp > top.end_time:
				retval = self.process_coincs(pad, inbuf)
				if retval != gst.FLOW_OK:
					return retval
				top = self.__queue.top

			for group in sngl_inspiral_groups_from_buffer(inbuf):
				# FIXME: Pick which sngl_inspiral field to hijack.
				# Currently I am using alpha to store per-detector IFAR.
				coinc = SnglCoinc(group, net_ifar((row.alpha for row in group), float(self.__dt)))
				if coinc.time > top.end_time:
					retval = self.process_coincs(pad, inbuf)
					if retval != gst.FLOW_OK:
						return retval
					top = self.__queue.top
				if coinc.ifar < self.__min_ifar:
					top.coinc_list.append(coinc)

			if inbuf.timestamp + inbuf.duration > top.end_time:
				retval = self.process_coincs(pad, inbuf)
				if retval != gst.FLOW_OK:
					return retval

			return gst.FLOW_OK

		except:
			self.error(traceback.format_exc())
			return gst.FLOW_ERROR


# Register element class
gstlal_element_register(lal_coincselector)
