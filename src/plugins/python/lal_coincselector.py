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
import traceback


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
		self.__srcpad.use_fixed_caps() # FIXME: better to use proxycaps
		self.__sinkpad.use_fixed_caps() # FIXME: better to use proxycaps
		self.__sinkpad.set_chain_function(self.chain)


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


	def chain(self, pad, inbuf):
		try: # FIXME: apparently the gst.Pad wrapper silences exceptions from chain() routines.
			# FIXME: Pick which sngl_inspiral field to hijack.
			# Currently I am using alpha to store per-detector IFAR.
			buf = sngl_inspiral_groups_to_buffer(
				(group for group in sngl_inspiral_groups_from_buffer(inbuf) if net_ifar((row.alpha for row in group), float(self.__dt)) >= float(self.__min_ifar)),
				inbuf.caps[0]['channels'])
			buf.offset = inbuf.offset
			buf.offset_end = inbuf.offset_end
			buf.timestamp = inbuf.timestamp
			buf.duration = inbuf.duration
			buf.caps = inbuf.caps
			# FIXME: copy flags too
			return self.__srcpad.push(inbuf)
		except:
			self.error(traceback.format_exc())
			return gst.FLOW_ERROR


# Register element class
gstlal_element_register(lal_coincselector)
