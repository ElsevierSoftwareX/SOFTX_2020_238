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


from matplotlib import cm as colourmap
import numpy


from gstlal.pipeutil import *
from gstlal import matplotlibhelper
from gstlal import pipeio
from gst.extend.pygobject import gproperty


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


class ArrayQueue(list):
	class Element(object):
		def __init__(self, buf):
			self.timestamp = buf.timestamp
			self.duration = buf.duration
			self.offset = buf.offset
			self.offset_end = buf.offset_end
			if bool(buf.flags & gst.BUFFER_FLAG_GAP):
				self.data = numpy.zeros((self.offset_end - self.offset, buf.caps[0]["channels"]), dtype = pipeio.numpy_dtype_from_caps(buf.caps))
			else:
				self.data = pipeio.array_from_audio_buffer(buf)

		def __len__(self):
			return self.data.shape[0]

		def flush(self, n):
			if n > self.data.shape[0]:
				n = self.data.shape[0]
			self.data = self.data[n:,:]
			delta_t = self.duration * float(n) / (self.offset_end - self.offset)
			self.timestamp += delta_t
			self.duration -= delta_t
			self.offset += n
			return n

		def get(self, n):
			return self.data[:n,:], self.timestamp

		def extract(self, n):
			extracted, timestamp = self.get(n)
			self.flush(n)
			return extracted, timestamp

	def append(self, buf):
		list.append(self, ArrayQueue.Element(buf))

	def __len__(self):
		if not list.__len__(self):
			return 0
		return self[-1].offset_end - self[0].offset

	def flush(self, n):
		flushed = n
		while n > 0 and self:
			n -= self[0].flush(n)
			if not len(self[0]):
				del self[0]
		return flushed - n

	def get(self, n):
		if self:
			timestamp = self[0].timestamp
		else:
			timestamp = None
		data = []
		for i in self:
			data.append(i.get(n)[0])
			n -= data[-1].shape[0]
			if not n:
				break
		return numpy.concatenate(data), timestamp

	def extract(self, n):
		extracted, timestamp = self.get(n)
		self.flush(extracted.shape[0])
		return extracted, timestamp


def yticks(min, max, n):
	delta = float(max - min) / n
	return sorted(set(min + int(round(delta * i)) for i in range(n + 1)))


class Channelgram(matplotlibhelper.BaseMatplotlibTransform):
	gproperty(
		gobject.TYPE_DOUBLE,
		"plot-width",
		"Width of the plot in seconds, 0 = 1/framerate",
		0.0,	# min
		gobject.G_MAXDOUBLE,	# max
		0.0,	# default
		readable = True, writable = True
	)

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-complex, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {64, 128};" +
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
		super(Channelgram, self).__init__()
		self.channels = None
		self.in_rate = None
		self.out_rate = None
		self.set_property("plot-width", 0.0)	# seconds, 0 = 1/framerate
		self.instrument = None
		self.channel_name = None
		self.sample_units = None


	def do_set_caps(self, incaps, outcaps):
		self.in_rate = incaps[0]["rate"]
		channels = incaps[0]["channels"]
		if channels != self.channels:
			self.queue = ArrayQueue()
		self.channels = channels
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


	def make_frame(self, samples, samples_timestamp, outbuf):
		#
		# set metadata and advance output offset counter
		#

		outbuf.offset = self.next_out_offset
		self.next_out_offset += 1
		outbuf.offset_end = self.next_out_offset
		outbuf.timestamp = self.t0 + int(round(float(int(outbuf.offset - self.offset0) / self.out_rate) * gst.SECOND))
		outbuf.duration = self.t0 + int(round(float(int(outbuf.offset_end - self.offset0) / self.out_rate) * gst.SECOND)) - outbuf.timestamp

		#
		# generate pseudocolor plot
		#

		axes = self.axes
		axes.clear()
		x, y = map(lambda n: numpy.arange(n + 1, dtype = "double"), samples.shape)
		x = x / self.in_rate + float(samples_timestamp) / gst.SECOND
		y -= 0.5
		xlim = x[0], x[-1]
		ylim = y[0], y[-1]
		x, y = numpy.meshgrid(x, y)
		if samples.dtype.kind == "c":
			#
			# complex data
			#

			axes.pcolormesh(x, y, numpy.abs(samples.transpose()), cmap = colourmap.gray)
		else:
			#
			# real data
			#

			axes.pcolormesh(x, y, samples.transpose(), cmap = colourmap.gray)
		axes.set_xlim(xlim)
		axes.set_ylim(ylim)
		axes.set_yticks(yticks(0, samples.shape[1] - 1, 20))
		axes.set_title(r"Amplitude of %s, %s" % (self.instrument or "Unknown Instrument", self.channel_name or "Unknown Channel"))
		axes.set_xlabel(r"Time (s)")
		axes.set_ylabel(r"Channel Number")

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
		# append input to queue
		#

		self.queue.append(inbuf)

		#
		# number of samples required for output frame, and the
		# number of samples flushed per output frame
		#
		# FIXME:  if the number flushed isn't really an integer the
		# output timestamps drift wrt the input timestamps
		#

		samples_flushed_per_frame = int(round(self.in_rate / float(self.out_rate)))
		plot_width = self.get_property("plot-width")
		if plot_width != 0.0:
			samples_per_frame = int(round(plot_width * self.in_rate))
		else:
			samples_per_frame = samples_flushed_per_frame

		#
		# build output frame(s)
		#

		if len(self.queue) < samples_per_frame:
			# not enough data for output
			# FIXME: should return
			# GST_BASE_TRANSFORM_FLOW_DROPPED, don't know what
			# that constant is, but I know it's #define'ed to
			# GST_FLOW_CUSTOM_SUCCESS.  figure out what the
			# constant should be
			return gst.FLOW_CUSTOM_SUCCESS

		while len(self.queue) >= 2 * samples_per_frame:
			flow_return, newoutbuf = self.get_pad("src").alloc_buffer(self.next_out_offset, outbuf.size, outbuf.caps)
			samples, timestamp = self.queue.get(samples_per_frame)
			self.queue.flush(samples_flushed_per_frame)
			self.get_pad("src").push(self.make_frame(samples, timestamp, newoutbuf))
		samples, timestamp = self.queue.get(samples_per_frame)
		self.queue.flush(samples_flushed_per_frame)
		self.make_frame(samples, timestamp, outbuf)

		#
		# done
		#

		return gst.FLOW_OK


	def do_transform_size(self, direction, caps, size, othercaps):
		samples_per_frame = int(round(self.in_rate / float(self.out_rate)))

		if direction == gst.PAD_SRC:
			#
			# convert byte count on src pad to sample count on
			# sink pad (minus samples we already have)
			#

			bytes_per_frame = caps[0]["width"] * caps[0]["height"] * caps[0]["bpp"] / 8
			samples = int(size / bytes_per_frame) * samples_per_frame - len(self.queue)

			#
			# convert to byte count on sink pad.
			#

			if samples <= 0:
				return 0
			return samples * (othercaps[0]["width"] // 8) * othercaps[0]["channels"]

		elif direction == gst.PAD_SINK:
			return super(Channelgram, self).do_transform_size(direction, caps, size, othercaps)


gobject.type_register(Channelgram)


def mkchannelgram(pipeline, src, **kwargs):
	elem = Channelgram()
	for name, value in kwargs.items():
		elem.set_property(name, value)
	pipeline.add(elem)
	src.link(elem)
	return elem
