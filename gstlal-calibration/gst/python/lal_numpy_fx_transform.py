# Copyright (C) 2015  Madeline Wade
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

"""
Arbitrary function generator based on sink pad timestamps and duration.
Accepts any Python expression.  The "numpy" module is available as if you 
typed "from numpy import *".  The local variable "t" provides the stream 
time in seconds.
"""
__author__ = "Madeline Wade <madeline.wade@ligo.org>"

import numpy
import gst
import sys
import gobject

from gstlal import pipeio
from gstlal.pipeutil import *

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def create_expression(inbuf, outbuf, caps, expression):
	rate = caps[0]["rate"]
	dt = 1.0/float(rate)
	t_start = float(inbuf.timestamp) / float(gst.SECOND)
	dur = float(inbuf.duration) / float(gst.SECOND)
	t_end = t_start + dur
	t = numpy.arange(t_start, t_end, dt)
	y = eval(expression, numpy.__dict__, {'t': t})

	unitsize = pipeio.get_unit_size(caps)
	bufsize = unitsize * len(t)
	outbuf[0:bufsize] = y.flatten().astype(pipeio.numpy_dtype_from_caps(caps)).data
	

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_numpy_fx_transform(gst.BaseTransform):
	__gstdetails__ = (
		"Arbitrary function generator from sink timestamps and duration",
		"Filter/Audio",
		__doc__,
		__author__
	)

	__gproperties__ = {
		'expression': (
			gobject.TYPE_STRING,
			'Expression',
			'any Python expression, to be evaluated under "from numpy import *"',
			'0 * t',
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
				"width = (int) 64"
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
				"width = (int) 64"
			)
		)
	)

	def __init__(self):
		super(lal_numpy_fx_transform, self).__init__()
		self.set_gap_aware(True)

	def do_set_property(self, prop, val):
		if prop.name == "expression":
			self.__compiled_expression = compile(val, "<compiled Python expression>", "eval")
			self.__expression = val

	def do_get_property(self, prop):
		if prop.name == "expression":
			return self.__expression
		
	def do_transform(self, inbuf, outbuf):
		pad = self.src_pads().next()
		caps = pad.get_caps()

		# FIXME: I'm not sure this is the right fix for hearbeat buffers, so I need to check this!
		if len(inbuf) == 0:
			gst.Buffer.flag_set(inbuf, gst.BUFFER_FLAG_GAP)

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			create_expression(inbuf, outbuf, caps, self.__expression)
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
	
		return gst.FLOW_OK

gobject.type_register(lal_numpy_fx_transform)

__gstelementfactory__ = (
	lal_numpy_fx_transform.__name__,
	gst.RANK_NONE,
	lal_numpy_fx_transform
)
