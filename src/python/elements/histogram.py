import matplotlib
matplotlib.rcParams.update({
	"font.size": 8.0,
	"axes.titlesize": 10.0,
	"axes.labelsize": 10.0,
	"xtick.labelsize": 8.0,
	"ytick.labelsize": 8.0,
	"legend.fontsize": 8.0,
	"figure.dpi": 100,
	"savefig.dpi": 100,
	"text.usetex": True,
	"path.simplify": True
})
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
import numpy


import gobject
import pygst
pygst.require('0.10')
import gst  


class Histogram(gst.BaseTransform):
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
				"video/x-raw-rgb, " +
				"width = (int) [1, MAX], " +
				"height = (int) [1, MAX], " +
				"framerate = (fraction) [0/1, MAX], " +
				"bpp = (int) 32, " +
				"depth = (int) 32, " +
				"red_mask = (int) -16777216, " +
				"green_mask = (int) 16711680, " +
				"blue_mask = (int) 65280, " +
				"alpha_mask = (int) 255, " +
				#"endianness = (int) BYTE_ORDER"
				"endianness = (int) 4321"
			)
		)
	)

	def __init__(self):
		gst.BaseTransform.__init__(self)
		self.in_rate = None
		self.out_rate = None
		self.out_width = None
		self.out_height = None
		self.buf = numpy.zeros((0,), dtype = "double")

	def do_set_caps(self, incaps, outcaps):
		self.in_rate = incaps[0]["rate"]
		self.out_rate = outcaps[0]["framerate"]
		self.out_width = outcaps[0]["width"]
		self.out_height = outcaps[0]["height"]
		return True

	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_out_offset = None
		return True

	def do_get_unit_size(self, caps):
		if caps[0].get_name() == "audio/x-raw-float":
			return caps[0]["channels"] * caps[0]["width"] / 8
		elif caps[0].get_name() == "video/x-raw-rgb":
			return caps[0]["width"] * caps[0]["height"] * caps[0]["bpp"] / 8
		else:
			raise ValueError, caps

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

		self.buf = numpy.append(self.buf, numpy.frombuffer(inbuf, dtype = "double"))

		#
		# number of samples required for output frame
		#

		samples_per_frame = int(round(self.in_rate / float(self.out_rate)))

		#
		# loop over output frames
		#

		frames = 0
		while True:
			if len(self.buf) < samples_per_frame:
				# not enough data for an output frame
				if not frames:
					# FIXME: should return
					# GST_BASE_TRANSFORM_FLOW_DROPPED,
					# don't know what that constant is,
					# but I know it's #define'ed to
					# GST_FLOW_CUSTOM_SUCCESS.  figure
					# out what the constant should be
					return gst.FLOW_CUSTOM_SUCCESS
				return gst.FLOW_OK

			#
			# generate the histogram
			#

			fig = figure.Figure()
			FigureCanvas(fig)
			fig.set_size_inches(self.out_width / float(fig.get_dpi()), self.out_height / float(fig.get_dpi()))
			axes = fig.gca(xlabel = "Amplitude", ylabel = "Count", title = "Histogram", rasterized = True)
			axes.hist(self.buf[:samples_per_frame], bins = 100)

			#
			# extract the pixel data
			#

			fig.canvas.draw()
			rgba_buffer = fig.canvas.buffer_rgba(0, 0)
			rgba_buffer_size = len(rgba_buffer)

			#
			# copy pixel data to output buffer and set metadata
			#

			outbuf[0:rgba_buffer_size] = rgba_buffer
			outbuf.timestamp = self.t0 + int(round(float((self.next_out_offset - self.offset0) / self.out_rate) * gst.SECOND))
			outbuf.offset = self.next_out_offset

			#
			# reset for next frame
			#

			self.buf = self.buf[samples_per_frame:]
			frames += 1
			self.next_out_offset += 1

	def do_transform_caps(self, direction, caps):
		if direction == gst.PAD_SRC:
			# convert src pad's caps to sink pad's
			return self.get_pad("sink").get_fixed_caps_func()
		elif direction == gst.PAD_SINK:
			# convert sink pad's caps to src pad's
			return self.get_pad("src").get_fixed_caps_func()
		raise ValueError

	def do_transform_size(self, direction, caps, size, othercaps):
		if direction == gst.PAD_SRC:
			# convert size on src pad to size on sink pad
			samples_per_frame = int(round(float(self.in_rate / self.out_rate)))
			if samples_per_frame <= len(self.buf):
				# don't need any more data to build a frame
				return 0
			# assume 8 bytes per input sample
			return (samples_per_frame - len(self.buf)) * 8
		elif direction == gst.PAD_SINK:
			# convert size on sink pad to size on src pad
			# assume 4 bytes per output pixel
			return self.out_width * self.out_height * 4
		raise ValueError, direction


gobject.type_register(Histogram)
