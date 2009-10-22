import matplotlib
matplotlib.rcParams.update({
	"font.size": 8.0,
	"axes.titlesize": 10.0,
	"axes.labelsize": 10.0,
	"xtick.labelsize": 8.0,
	"ytick.labelsize": 8.0,
	"legend.fontsize": 8.0,
	"figure.dpi": 200,
	"savefig.dpi": 200,
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
				"width = (int) 640, " +
				"height = (int) 480, " +
				"framerate = (fraction) 1/10, " +
				"bpp = (int) 32, " +
				"depth = (int) 24, " +
				"endianness = (int) BYTE_ORDER"
			)
		)
	)


	def __init__(self):
		gst.BaseTransform.__init__(self)
		self.in_rate = None
		self.out_rate = None
		self.out_width = None
		self.out_height = None
		self.fig = None
		self.buf = numpy.zeros((0,), dtype = "double")

	def do_set_caps(self, incaps, outcaps):
		self.in_rate = incaps[0]["rate"]
		self.out_rate = outcaps[0]["framerate"]
		self.out_width = outcaps[0]["width"]
		self.out_height = outcaps[0]["height"]
		self.t0 = None
		self.offset0 = None
		self.next_out_offset = None
		return True

	def do_start(self):
		self.fig = figure.Figure()
		FigureCanvas(self.fig)
		self.fig.set_size_inches(self.out_width / float(self.fig.get_dpi()), self.out_height / float(self.fig.get_dpi()))

	def do_stop(self):
		self.fig = None

	def do_get_unit_size(self, caps):
		if caps[0].get_name() == "audio/x-raw-float":
			return caps[0]["channels"] * caps[0]["width"] / 8
		elif caps[0].get_name() == "video/x-raw-rgb":
			return caps[0]["width"] * caps[0]["height"] * caps[0]["bpp"] / 8
		else:
			raise ValueError, caps

	def do_transform(self, inbuf, outbuf):
		# update metadata
		if self.t0 is None:
			self.t0 = inbuf.timestamp
			self.offset0 = 0
			self.next_out_offset = 0

		# append input to storage buffer
		self.buf.append(numpy.frombuffer(inbuf.data, dtype = "double"))

		# number of samples required for output frame
		N = int(round(self.in_rate / float(self.out_rate)))

		# loop over output frames
		frames = 0
		while True:
			if len(self.buf) < N:
				# not enough data for an output frame
				if not frames:
					# FIXME: should return
					# GST_BASE_TRANSFORM_FLOW_DROPPED,
					# don't know what that constant is,
					# but I know it's #define'ed to
					# GST_FLOW_CUSTOM_SUCCESS.  figure
					# out what the constant should be
					return gst.FLOW_CUSTOM_SUCCESS
				return gst.FLOW_SUCCESS

			axes = self.fig.gca(xlabel = "Amplitude", ylabel = "Count", title = "Histogram", rasterized = True)
			axes.hist(self.buf[:N], bins = 100)

			outdata = numpy.frombuffer(outbuf.data, dtype = "uint32")
			outdata[:] = numpy.zeros((len(outdata),), dtype = "uint32")
			outbuf.timestamp = self.t0 + int(round((self.next_out_offset - self.offset0) / float(self.out_rate) * gst.SECOND))
			outbuf.offset = self.next_out_offset

			self.fig.clear()
			del self.buf[:N]
			frames += 1
			self.next_out_offset += 1

	def do_transform_size(self, direction, caps, size, othercaps):
		if direction == gst.PAD_SRC:
			# assume 4 bytes per output pixel
			return self.out_width * self.out_height * 4
		# direction == gst.PAD_SINK
		samples_per_frame = int(self.in_rate / float(self.out_rate))
		if samples_per_frame <= len(self.buf):
			# don't need any more data to build a frame
			return 0
		# assume 8 bytes per input sample
		return (samples_per_frame - len(self.buf)) * 8


gobject.type_register(Histogram)
