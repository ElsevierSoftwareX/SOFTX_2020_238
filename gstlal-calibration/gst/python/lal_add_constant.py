# Copyright (C) 2015  Madeline Wade, Chris Pankow
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

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

__author__ = "Madeline Wade <madeline.wade@ligo.org>"

import numpy
import gst
import sys
import gobject

from gstlal import pipeio

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def add_constant_value(inbuf, outbuf, constant):
	out = []
	current = numpy.frombuffer(inbuf[:], dtype = numpy.float64)
	for i in current:
		val = i + constant
		out.append(val)
	out = numpy.array(out, dtype = numpy.float64)
	out_len = out.nbytes
	outbuf[:out_len] = numpy.getbuffer(out)

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_add_constant(gst.BaseTransform):
	__gstdetails__ = (
		"Add a constant value",
		"Filter/Audio",
		"Adds a constant value to a data stream",
		__author__
	)

	__gproperties__ = {
		"constant" : (
			gobject.TYPE_DOUBLE,
			"Constant",
			"Constant to be added to stream",
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, 0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64 "
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64 "
			)
		)
	)

	def __init__(self):
		super(lal_add_constant, self).__init__()
		self.set_gap_aware(True)

	def do_set_property(self, prop, val):
		if prop.name == "constant":
			self.constant = val

	def do_get_property(self, prop):
		if prop.name == "constant":
			return self.constant
	
	def do_start(self):
		self.constant = 0.0
		return True
	
	def do_transform(self, inbuf, outbuf):
		# FIXME: I'm not sure this is the right fix for hearbeat buffers, so I need to check this!
		if len(inbuf) == 0:
			gst.Buffer.flag_set(inbuf, gst.BUFFER_FLAG_GAP)

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			add_constant_value(inbuf, outbuf, self.constant) 
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
	
		return gst.FLOW_OK

gobject.type_register(lal_add_constant)

__gstelementfactory__ = (
	lal_add_constant.__name__,
	gst.RANK_NONE,
	lal_add_constant
)
