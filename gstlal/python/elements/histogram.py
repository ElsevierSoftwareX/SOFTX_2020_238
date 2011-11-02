# Copyright (C) 2009  Kipp Cannon
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


import numpy


from gstlal.pipeutil import *
from gstlal import matplotlibhelper
from gstlal import pipeio


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


class Histogram(matplotlibhelper.BaseMatplotlibTransform):
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {32, 64};" +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32," +
				"depth = (int) 32," +
				"signed = (bool) {true, false}; " +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64," +
				"depth = (int) 64," +
				"signed = (bool) {true, false}"
			)
		),
		matplotlibhelper.BaseMatplotlibTransform.__gsttemplates__
	)


	def __init__(self):
		super(Histogram, self).__init__()
		self.channels = None
		self.in_rate = None
		self.out_rate = None
		self.instrument = None
		self.channel_name = None
		self.sample_units = None


	def do_set_caps(self, incaps, outcaps):
		channels = incaps[0]["channels"]
		if channels != self.channels:
			self.buf = numpy.zeros((0, channels), dtype = pipeio.numpy_dtype_from_caps(incaps))
		self.channels = channels
		self.in_rate = incaps[0]["rate"]
		self.out_rate = outcaps[0]["framerate"]
		return True


	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_out_offset = None
		return True


	def do_event(self, event):
		if event.type == gst.EVENT_TAG:
			tags = pipeio.parse_framesrc_tags(event.parse_tag())
			self.instrument = tags["instrument"]
			self.channel_name = tags["channel-name"]
			self.sample_units = tags["sample-units"]
		return True


	def make_frame(self, samples, outbuf):
		#
		# set metadata and advance output offset counter
		#

		outbuf.offset = self.next_out_offset
		self.next_out_offset += 1
		outbuf.offset_end = self.next_out_offset
		outbuf.timestamp = self.t0 + int(round(float(int(outbuf.offset - self.offset0) / self.out_rate) * gst.SECOND))
		outbuf.duration = self.t0 + int(round(float(int(outbuf.offset_end - self.offset0) / self.out_rate) * gst.SECOND)) - outbuf.timestamp

		#
		# generate histogram
		#

		axes = self.axes
		axes.clear()
		for channel in numpy.transpose(samples)[:]:
			axes.hist(channel, bins = 101, histtype = "step")
		axes.set_yscale("log")
		axes.grid(True)
		axes.set_xlabel(r"Amplitude (%s)" % ((self.sample_units is not None) and (str(self.sample_units) or "dimensionless") or "unkown units"))
		axes.set_ylabel(r"Count")
		axes.set_title(r"%s, %s (%.9g s -- %.9g s)" % (self.instrument or "Unknown Instrument", self.channel_name or "Unknown Channel", float(outbuf.timestamp) / gst.SECOND, float(outbuf.timestamp + outbuf.duration) / gst.SECOND))

		#
		# copy pixel data to output buffer
		#

		matplotlibhelper.render(self.figure, outbuf)

		#
		# done
		#

		return outbuf


	def do_transform(self, inbuf, outbuf):
		#
		# make sure we have valid metadata
		#

		if self.t0 is None:
			self.t0 = inbuf.timestamp
			self.offset0 = 0
			self.next_out_offset = 0

		#
		# append input to time series buffer
		#

		self.buf = numpy.concatenate((self.buf, pipeio.array_from_audio_buffer(inbuf)))

		#
		# number of samples required for output frame
		#

		samples_per_frame = int(round(self.in_rate / float(self.out_rate)))

		#
		# build output frame(s)
		#

		if len(self.buf) < samples_per_frame:
			# not enough data for output
			# FIXME: should return
			# GST_BASE_TRANSFORM_FLOW_DROPPED, don't know what
			# that constant is, but I know it's #define'ed to
			# GST_FLOW_CUSTOM_SUCCESS.  figure out what the
			# constant should be
			return gst.FLOW_CUSTOM_SUCCESS

		while len(self.buf) >= 2 * samples_per_frame:
			flow_return, newoutbuf = self.get_pad("src").alloc_buffer(self.next_out_offset, outbuf.size, outbuf.caps)
			self.get_pad("src").push(self.make_frame(self.buf[:samples_per_frame], newoutbuf))
			self.buf = self.buf[samples_per_frame:]
		self.make_frame(self.buf[:samples_per_frame], outbuf)
		self.buf = self.buf[samples_per_frame:]

		#
		# done
		#

		return gst.FLOW_OK


	def do_transform_size(self, direction, caps, size, othercaps):
		samples_per_frame = int(round(float(self.in_rate / self.out_rate)))

		if direction == gst.PAD_SRC:
			#
			# convert byte count on src pad to sample count on
			# sink pad (minus samples we already have)
			#

			bytes_per_frame = caps[0]["width"] * caps[0]["height"] * caps[0]["bpp"] / 8
			samples = int(size / bytes_per_frame) * samples_per_frame - len(self.buf)

			#
			# convert to byte count on sink pad.
			#

			if samples <= 0:
				return 0
			return samples * (othercaps[0]["width"] // 8) * othercaps[0]["channels"]

		else:
			return super(Histogram, self).do_transform_size(direction, caps, size, othercaps)


gobject.type_register(Histogram)


def mkhistogram(pipeline, src):
	elem = Histogram()
	pipeline.add(elem)
	src.link(elem)
	return elem
