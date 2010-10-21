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
Determine nongap duty cycle of a stream
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


import sys
from gstlal.pipeutil import *


class lal_nongap_duty_cycle(gst.BaseTransform):
	__gstdetails__ = (
		'Nongap Duty Cycle',
		'Generic',
		__doc__,
		__author__
	)

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string("audio/x-raw-float")
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string("audio/x-raw-float")
		)
	)

	def __init__(self):
		super(lal_nongap_duty_cycle, self).__init__()
		self.set_gap_aware(True)
		self.set_passthrough(True)
		self.set_in_place(True)

	def reset_state(self):
		self.nongap_samples = 0L
		self.total_samples = 0L

	def do_start(self):
		self.reset_state()
		return True

	def do_transform_ip(self, buf):
		if buf.flag_is_set(gst.BUFFER_FLAG_DISCONT):
			self.reset_state()

		samples = buf.offset_end - buf.offset
		if not buf.flag_is_set(gst.BUFFER_FLAG_GAP):
			self.nongap_samples += samples
		self.total_samples += samples

		print >>sys.stderr, "%s: duty cycle = %.2f" % (self.get_property("name"), 100. * self.nongap_samples / self.total_samples)
		return gst.FLOW_OK


# Register element class
gstlal_element_register(lal_nongap_duty_cycle)
