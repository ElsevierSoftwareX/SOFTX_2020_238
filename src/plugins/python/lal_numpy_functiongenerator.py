# Copyright (C) 2010  Leo Singer
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
Arbitrary function generator.  Accepts any Python expression.  The "numpy"
module is available as if you typed "from numpy import *".  The local variable
"t" provides the stream time in seconds.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gstlal import pipeio
import numpy


# TODO: In order to prepare for submission to GStreamer ...
# Try using "math" instead of "numpy", and evaluating the expression
# in a loop, in order to eliminate the Numpy dependency.


class lal_numpy_functiongenerator(gst.BaseSrc):

	__gstdetails__ = (
		"Arbitrary function generator",
		"Source",
		__doc__,
		__author__
	)
	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				audio/x-raw-float,
				channels = (int) 1,
				endianness = (int) BYTE_ORDER,
				width = (int) 64;
				audio/x-raw-complex,
				channels = (int) 1,
				endianness = (int) BYTE_ORDER,
				width = (int) 128
			""")
		),
	)
	__gproperties__ = {
		'expression': (
			gobject.TYPE_STRING,
			'Expression',
			'any Python expression, to be evaluated under "from numpy import *"',
			"0 * t",
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'samplesperbuffer': (
			gobject.TYPE_INT,
			'Samples per buffer',
			'Number of samples in each outgoing buffer',
			1, gobject.G_MAXINT, 1024,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}


	def __init__(self):
		super(lal_numpy_functiongenerator, self).__init__()
		self.set_do_timestamp(False)
		self.set_format(gst.FORMAT_TIME)
		self.src_pads().next().use_fixed_caps()
		self.__start_time = 0
		self.__last_offset_end = 0


	def do_set_property(self, prop, val):
		"""gobject->set_property virtual method."""
		if prop.name == 'expression':
			self.__compiled_expression = compile(val, "<compiled Python expression>", "eval")
			self.__expression = val
		elif prop.name == 'samplesperbuffer':
			self.__samplesperbuffer = val


	def do_get_property(self, prop):
		"""gobject->get_property virtual method."""
		if prop.name == 'expression':
			return self.__expression
		elif prop.name == 'samplesperbuffer':
			return self.__samplesperbuffer


	def do_check_get_range(self):
		"""GstBaseSrc->check_get_range virtual method"""
		return True


	def do_is_seekable(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return True


	def do_seek(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return True


	def do_do_seek(self, segment):
		"""GstBaseSrc->do_seek virtual method"""
		if segment.format != gst.FORMAT_TIME:
			self.error("lal_numpy_functiongenerator only supports GST_FORMAT_TIME")
			return False
		else:
			self.__start_time = segment.start
			self.__last_offset_end = 0
			return True


	def do_create(self, offset, size):
		"""GstBaseSrc->create virtual method"""

		# Look up the pad, caps, and sample rate
		pad = self.src_pads().next()
		caps = pad.get_caps()
		rate = caps[0]["rate"]

		# Evalluate expression
		t = (self.__last_offset_end + numpy.arange(self.__samplesperbuffer)) / float(rate) + self.__start_time / float(gst.SECOND)
		y = eval(self.__compiled_expression, numpy.__dict__, {'t': t})

		# Allocate output buffer
		unitsize = pipeio.get_unit_size(caps)
		bufsize = unitsize * self.__samplesperbuffer
		(retval, buf) = pad.alloc_buffer(self.__last_offset_end, bufsize, caps)
		if retval != gst.FLOW_OK:
			return (retval, None)

		# Copy data from Numpy array
		buf[0:bufsize] = y.flatten().astype(pipeio.numpy_dtype_from_caps(caps)).data

		# Set buffer metadata and update our sample counters
		buf.timestamp = long(gst.SECOND * self.__last_offset_end / float(rate) + self.__start_time)
		buf.duration = long(self.__samplesperbuffer * gst.SECOND / float(rate))
		buf.offset = self.__last_offset_end
		self.__last_offset_end += self.__samplesperbuffer
		buf.offset_end = self.__last_offset_end

		# Done!
		return (gst.FLOW_OK, buf)


# Register element class
gstlal_element_register(lal_numpy_functiongenerator)
