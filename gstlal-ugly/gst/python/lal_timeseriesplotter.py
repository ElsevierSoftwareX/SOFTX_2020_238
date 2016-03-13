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


class lal_timeseriesplotter(matplotlibhelper.BaseMatplotlibTransform):


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


	def do_set_property(self, prop, val):
		"""gobject->set_property virtual method."""
		if prop.name == 'y-autoscale':
			self.axes.set_autoscaley_on(val)
		elif prop.name == 'y-min':
			self.axes.set_ylim(val, self.axes.get_ylim()[1])
		elif prop.name == 'y-max':
			self.axes.set_ylim(self.axes.get_ylim()[0], val)


	def do_get_property(self, prop):
		"""gobject->get_property virtual method."""
		if prop.name == 'y-autoscale':
			return self.axes.get_autoscaley_on()
		elif prop.name == 'y-min':
			return self.axes.get_ylim()[0]
		elif prop.name == 'y-max':
			return self.axes.get_ylim()[1]


	def do_set_caps(self, incaps, outcaps):
		self.size = outcaps['width'], outcaps['height']
		self.fmt = outcaps['format']


	def do_transform(self, inbuf, outbuf):
		"""GstBaseTransform->transform virtual method."""

		# Convert received buffer to Numpy array.
		data = pipeio.array_from_audio_buffer(inbuf)

		# Build plot.
		self.axes.plot(data, 'k')
		self.axes.set_xlim(0, len(data))
		self.axes.set_xlabel("samples since %d.%08d" % (inbuf.timestamp / gst.SECOND, inbuf.timestamp % gst.SECOND))

		# Render to output buffer.
		matplotlibhelper.render(self.figure, outbuf, self.size, self.fmt)

		# Erase old lines.
		self.axes.lines = []
		self.axes.xaxis.cla() # FIXME: Matplotlib leaks callback refs without this
		self.axes.yaxis.cla() # FIXME: Matplotlib leaks callback refs without this

		# Copy timing information to output buffer.
		outbuf.timestamp = inbuf.timestamp
		outbuf.duration = inbuf.duration
		outbuf.offset = inbuf.offset
		outbuf.offset_end = inbuf.offset_end

		# Done!
		return gst.FLOW_OK


# Register element class
gstlal_element_register(lal_timeseriesplotter)
