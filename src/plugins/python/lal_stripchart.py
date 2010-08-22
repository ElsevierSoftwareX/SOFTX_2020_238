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
Time series plotter.  Renders time series from any number of input streams.
This element demonstrates how to recycle a Figure object in order to render
a sequence of similar plots very rapidly.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gstlal import pipeio
from gstlal import matplotlibhelper
from matplotlib.transforms import Bbox
import numpy


def array_from_audio_buffer(buf, caps):
	channels = caps[0]["channels"]
	a = numpy.frombuffer(buf, dtype = pipeio.numpy_dtype_from_caps(caps))
	return a.reshape((len(a) / channels, channels))


class lal_stripchart(matplotlibhelper.BaseMatplotlibTransform):


	__gstdetails__ = (
		"Time series renderer",
		"Filter",
		__doc__,
		__author__
	)
	__gproperties__ = {
		'y-autoscale': (
			gobject.TYPE_BOOLEAN,
			'y-autoscale',
			'If TRUE, then y-axis will be automatically scaled to fit data.',
			False,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'y-min': (
			gobject.TYPE_DOUBLE,
			'y-min',
			'Lower limit of y-axis',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, -2.0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'y-max': (
			gobject.TYPE_DOUBLE,
			'y-max',
			'Upper limit of y-axis',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, 2.0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'samplesperbuffer': (
			gobject.TYPE_INT,
			'samplesperbuffer',
			'Samples per input buffer',
			1, gobject.G_MAXINT, 1024,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'title': (
			gobject.TYPE_STRING,
			'title',
			'Title of plot',
			None,
			gobject.PARAM_WRITABLE
		)
	}
	__gsttemplates__ = (
		matplotlibhelper.BaseMatplotlibTransform.__gsttemplates__,
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				audio/x-raw-float,
				endianness = (int) BYTE_ORDER,
				width = (int) {32, 64},
				channels = (int) 1
			""")
		)
	)


	def __init__(self):
		super(lal_stripchart, self).__init__()
		self.line2D = self.axes.plot(numpy.zeros(1024))[0]


	def do_start(self):
		"""GstBaseTransform->start virtual method."""
		self.__adapter = gst.Adapter()
		self.__last_end_time = 0
		self.__last_offset_end = 0
		return True


	def do_stop(self):
		del self.__adapter
		return True


	def do_set_property(self, prop, val):
		"""gobject->set_property virtual method."""
		if prop.name == 'y-autoscale':
			self.__y_autoscale = val
		elif prop.name == 'y-min':
			self.axes.set_ylim(val, self.axes.get_ylim()[1])
		elif prop.name == 'y-max':
			self.axes.set_ylim(self.axes.get_ylim()[0], val)
		elif prop.name == 'title':
			self.axes.set_title(val)
		elif prop.name == 'samplesperbuffer':
			self.axes.lines = []
			self.axes.xaxis.cla() # FIXME: Matplotlib leaks callback refs without this
			self.axes.yaxis.cla() # FIXME: Matplotlib leaks callback refs without this
			self.line2D = self.axes.plot(numpy.zeros(val))[0]
			self.axes.set_xlim(0, val)


	def do_get_property(self, prop):
		"""gobject->get_property virtual method."""
		if prop.name == 'y-autoscale':
			return self.__y_autoscale
		elif prop.name == 'y-min':
			return self.axes.get_ylim()[0]
		elif prop.name == 'y-max':
			return self.axes.get_ylim()[1]
		elif prop.name == 'samplesperbuffer':
			return len(self.line2D.get_ydata())


	def do_set_caps(self, incaps, outcaps):
		self.__incaps = incaps
		self.__outcaps = outcaps
		self.__in_t0 = gst.CLOCK_TIME_NONE
		self.__in_offset0 = gst.BUFFER_OFFSET_NONE
		self.__last_in_offset_end = gst.BUFFER_OFFSET_NONE
		self.__last_out_offset_end = gst.BUFFER_OFFSET_NONE
		self.__in_rate = incaps[0]["rate"]
		self.__in_unit_size = pipeio.get_unit_size(incaps)
		self.__framerate = outcaps[0]["framerate"]
		return True


	def __last_offset_to_pop(self):
		return gst.util_uint64_scale(
			self.__last_out_offset_end,
			self.__framerate.denom * self.__in_rate,
			self.__framerate.num) + self.__in_offset0


	def __render(self, buf, samples_popped):
		matplotlibhelper.render(self.figure, buf)
		buf.offset = self.__last_out_offset_end
		self.__last_out_offset_end += 1
		buf.offset_end = self.__last_out_offset_end
		buf.timestamp = gst.util_uint64_scale(self.__last_in_offset_end - self.__in_offset0, gst.SECOND, self.__in_rate) + self.__in_t0
		self.__last_in_offset_end += samples_popped
		buf.duration = gst.util_uint64_scale(self.__last_in_offset_end - self.__in_offset0, gst.SECOND, self.__in_rate) + self.__in_t0 - buf.timestamp


	def do_transform(self, inbuf, outbuf):
		"""GstBaseTransform->transform virtual method."""

		samplesperbuffer = self.get_property("samplesperbuffer")

		if self.__in_t0 == gst.CLOCK_TIME_NONE:
			self.__in_t0 = inbuf.timestamp
			self.__in_offset0 = inbuf.offset
			self.__last_in_offset_end = inbuf.offset
			self.__last_out_offset_end = 0

		self.__adapter.push(inbuf)

		last_offset_to_pop = self.__last_offset_to_pop()
		if last_offset_to_pop <= self.__last_in_offset_end:
			self.__last_out_offset_end += 1
			return gst.FLOW_CUSTOM_SUCCESS
		new_samples_to_pop = last_offset_to_pop - self.__last_in_offset_end
		inbuf = self.__adapter.take_buffer(new_samples_to_pop * self.__in_unit_size)
		if inbuf is None:
			return gst.FLOW_CUSTOM_SUCCESS

		while True:
			samples_to_pop = new_samples_to_pop
			if samples_to_pop > samplesperbuffer:
				inbuf = inbuf.create_sub((samples_to_pop - samplesperbuffer) * self.__in_unit_size, samplesperbuffer * self.__in_unit_size)
			data = array_from_audio_buffer(inbuf, self.__incaps).flatten()
			data = numpy.concatenate( (self.line2D.get_ydata()[len(data):], data) )
			if self.__y_autoscale:
				min = data.min()
				max = data.max()
				diff = max - min
				self.axes.set_ylim(min - 0.1 * diff, max + 0.1 * diff)
			self.line2D.set_ydata(data)

			last_offset_to_pop = self.__last_offset_to_pop()
			if last_offset_to_pop <= self.__last_in_offset_end:
				self.__last_out_offset_end += 1
				break
			new_samples_to_pop = last_offset_to_pop - self.__last_in_offset_end
			inbuf = self.__adapter.take_buffer(new_samples_to_pop * self.__in_unit_size)
			if inbuf is None:
				break

			buf = gst.buffer_new_and_alloc(outbuf.size)
			buf.caps = outbuf.caps
			self.__render(buf, samples_to_pop)
			retval = self.src_pads().next().push(buf)
			if retval != gst.FLOW_OK:
				return retval

		self.__render(outbuf, samples_to_pop)
		return gst.FLOW_OK


# Register element class
gstlal_element_register(lal_stripchart)
