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
Fake skymap generator.  Creates random output that mimics the "lal_skymap"
element.  Buffers consist of 4 channels of double precision values.

channel 0: theta: geographic eleveation angle (colatitude) in radians, [0, pi)
channel 1:   phi: geographic azimuthal angle (longitude) in radians, [0, 2 pi)
channel 2:  span: size in radians of each side of the square pixel
channel 3:  logp: log probability
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gst.extend.pygobject import gproperty, with_construct_properties
import numpy


class lal_fakeskymapsrc(gst.BaseSrc):

	__gstdetails__ = (
		"Fake skymap source",
		"Source",
		__doc__,
		__author__
	)
	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				audio/x-raw-float,
				channels = (int) 4,
				endianness = (int) BYTE_ORDER,
				width = (int) 64
			""")
		),
	)


	def __init__(self):
		super(lal_fakeskymapsrc, self).__init__()
		self.set_do_timestamp(False)
		self.set_format(gst.FORMAT_TIME)
		self.src_pads().next().use_fixed_caps()


	def do_start(self):
		"""GstBaseSrc->start virtual method"""

		self.__last_offset_end = 0
		self.__last_time_end = 0

		return True


	def do_check_get_range(self):
		"""GstBaseSrc->check_get_range virtual method"""
		return False


	def do_is_seekable(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return False


	def do_create(self, offset, size):
		"""GstBaseSrc->create virtual method"""

		# Look up our src pad
		pad = self.src_pads().next()

		# Create a random number of random pixels
		npixels = numpy.random.randint(20000)
		theta = numpy.random.uniform(0.0, numpy.pi, (1, npixels))
		phi = numpy.random.uniform(0.0, 2*numpy.pi, (1, npixels))
		span = numpy.random.uniform(0.0, 1.0, (1, npixels)) # FIXME what is a typical pixel span?
		logp = numpy.random.uniform(-5.0, 5.0, (1, npixels))

		# Concatenate all the arrays
		skymap_array = numpy.hstack( (theta, phi, span, logp) )

		# Get raw binary data from Numpy array
		skymap_buffer = skymap_array.data

		# Allocate a new buffer
		(retval, buf) = pad.alloc_buffer(npixels, len(skymap_buffer), pad.get_property("caps"))

		# If pad.alloc_buffer failed, complain about it and stop
		if retval != gst.FLOW_OK:
			return (retval, None)

		# Set buffeer metadata
		buf.timestamp = self.__last_time_end
		buf.duration = gst.CLOCK_TIME_NONE
		buf.offset = self.__last_offset_end
		self.__last_offset_end += npixels
		buf.offset_end = self.__last_offset_end

		# Copy Numpy data into buffer
		buf[0:len(skymap_buffer)] = skymap_buffer

		# Done!  Send the buffer on its way.
		return (gst.FLOW_OK, buf)


# Register element class
gstlal_element_register(lal_fakeskymapsrc)
