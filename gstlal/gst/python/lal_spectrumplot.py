# Copyright (C) 2009--2011  Kipp Cannon
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


import bisect
import math
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
from matplotlib import cm as colourmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy


import pygtk
pygtk.require("2.0")
import gobject
import pygst
pygst.require('0.10')
import gst


from gstlal import pipeio
from gstlal.elements import matplotlibcaps


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


class lal_spectrumplot(gst.BaseTransform):
	__gstdetails__ = (
		"Power spectrum plot",
		"Plots",
		"Generates a video showing a power spectrum (e.g., as measured by lal_whiten)",
		__author__
	)

	__gproperties__ = {
		"f-min": (
			gobject.TYPE_DOUBLE,
			"f_{min}",
			"Lower bound of plot in Hz.",
			0, gobject.G_MAXDOUBLE, 10.0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"f-max": (
			gobject.TYPE_DOUBLE,
			"f_{max}",
			"Upper bound of plot in Hz.",
			0, gobject.G_MAXDOUBLE, 4000.0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"delta-f = (double) [0, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"rate = (fraction) [0/1, 2147483647/1], " +
				"width = (int) 64"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				matplotlibcaps + ", " +
				"width = (int) [1, MAX], " +
				"height = (int) [1, MAX], " +
				"framerate = (fraction) [0/1, 2147483647/1]"
			)
		)
	)


	def __init__(self):
		gst.BaseTransform.__init__(self)
		self.get_pad("sink").use_fixed_caps()
		self.get_pad("src").use_fixed_caps()
		self.channels = None
		self.delta_f = None
		self.out_width = 320	# default
		self.out_height = 200	# default
		self.instrument = None
		self.channel_name = None
		self.sample_units = None


	def do_set_property(self, prop, val):
		if prop.name == "f-min":
			self.f_min = val
		elif prop.name == "f-max":
			self.f_max = val
		else:
			raise AssertError


	def do_get_property(self, prop, val):
		if prop.name == "f-min":
			return self.f_min
		elif prop.name == "f-max":
			return self.f_max
		else:
			raise AssertError


	def do_set_caps(self, incaps, outcaps):
		self.channels = incaps[0]["channels"]
		self.delta_f = incaps[0]["delta-f"]
		self.out_width = outcaps[0]["width"]
		self.out_height = outcaps[0]["height"]
		return True


	def do_get_unit_size(self, caps):
		return pipeio.get_unit_size(caps)


	def do_event(self, event):
		if event.type == gst.EVENT_TAG:
			tags = pipeio.parse_framesrc_tags(event.parse_tag())
			self.instrument = tags["instrument"]
			self.channel_name = tags["channel-name"]
			self.sample_units = tags["sample-units"]
		return True


	def do_transform(self, inbuf, outbuf):
		#
		# generate spectrum plot
		#

		fig = figure.Figure()
		FigureCanvas(fig)
		fig.set_size_inches(self.out_width / float(fig.get_dpi()), self.out_height / float(fig.get_dpi()))
		axes = fig.gca(rasterized = True)

		data = numpy.transpose(pipeio.array_from_audio_buffer(inbuf))
		f = numpy.arange(len(data[0]), dtype = "double") * self.delta_f

		imin = bisect.bisect_left(f, self.f_min)
		imax = bisect.bisect_right(f, self.f_max)

		for psd in data[:]:
			axes.loglog(f[imin:imax], psd[imin:imax], alpha = 0.7)

		axes.grid(True)
		axes.set_xlim((self.f_min, self.f_max))
		axes.set_title(r"Spectral Density of %s, %s at %.11g s" % (self.instrument or "Unknown Instrument", self.channel_name or "Unknown Channel", float(inbuf.timestamp) / gst.SECOND))
		axes.set_xlabel(r"Frequency (Hz)")
		axes.set_ylabel(r"Spectral Density (%s)" % self.sample_units)

		#
		# extract pixel data
		#

		fig.canvas.draw()
		rgba_buffer = fig.canvas.buffer_rgba(0, 0)
		rgba_buffer_size = len(rgba_buffer)

		#
		# copy pixel data to output buffer
		#

		outbuf[0:rgba_buffer_size] = rgba_buffer
		outbuf.datasize = rgba_buffer_size

		#
		# set metadata on output buffer
		#

		outbuf.offset_end = outbuf.offset = gst.BUFFER_OFFSET_NONE
		outbuf.timestamp = inbuf.timestamp
		outbuf.duration = gst.CLOCK_TIME_NONE

		#
		# done
		#

		return gst.FLOW_OK


	def do_transform_caps(self, direction, caps):
		if direction == gst.PAD_SRC:
			#
			# convert src pad's caps to sink pad's
			#

			rate, = [struct["framerate"] for struct in caps]
			result = gst.Caps()
			for struct in self.get_pad("sink").get_pad_template_caps():
				struct = struct.copy()
				struct["rate"] = rate
				result.append_structure(struct)
			return result

		elif direction == gst.PAD_SINK:
			#
			# convert sink pad's caps to src pad's
			#

			rate, = [struct["rate"] for struct in caps]
			result = gst.Caps()
			for struct in self.get_pad("src").get_pad_template_caps():
				struct = struct.copy()
				struct["framerate"] = rate
				result.append_structure(struct)
			return result

		raise ValueError, direction


	def do_transform_size(self, direction, caps, size, othercaps):
		if direction == gst.PAD_SRC:
			#
			# compute frame count on src pad
			#

			frames = size * 8 // (caps[0]["bpp"] * caps[0]["width"] * caps[0]["height"])

			#
			# if greater than 1, ask for 1 byte.  lal_whiten can
			# only provide whole PSD buffer, so any non-zero
			# amount should produce a full PSD.  and lal_whiten
			# only operates in push mode so this is a non-issue
			#

			if frames < 1:
				return 0
			return 1

		elif direction == gst.PAD_SINK:
			#
			# any buffer on sink pad is turned into exactly
			# one frame on source pad
			#

			# FIXME:  figure out whats wrong with this
			# function, why is othercaps not right!?
			othercaps = self.get_pad("src").get_allowed_caps()
			return othercaps[0]["width"] * othercaps[0]["height"] * othercaps[0]["bpp"] // 8

		raise ValueError, direction


#
# register element class
#


gobject.type_register(lal_spectrumplot)

__gstelementfactory__ = (
	lal_spectrumplot.__name__,
	gst.RANK_NONE,
	lal_spectrumplot
)
