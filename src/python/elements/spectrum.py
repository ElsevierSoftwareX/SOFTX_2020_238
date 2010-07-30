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


import bisect
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


class Spectrum(matplotlibhelper.BaseMatplotlibTransform):
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"delta-f = (double) [0, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		),
		matplotlibhelper.BaseMatplotlibTransform.__gsttemplates__
	)


	def __init__(self):
		super(Spectrum, self).__init__()
		self.channels = None
		self.delta_f = None
		self.instrument = None
		self.channel_name = None
		self.sample_units = None


	def do_set_caps(self, incaps, outcaps):
		self.channels = incaps[0]["channels"]
		self.delta_f = incaps[0]["delta-f"]
		return True


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

		axes = self.axes
		axes.clear()

		data = numpy.transpose(pipeio.array_from_audio_buffer(inbuf))
		f = numpy.arange(len(data[0]), dtype = "double") * self.delta_f

		fmin, fmax = 30.0, 3000.0
		imin = bisect.bisect_left(f, fmin)
		imax = bisect.bisect_right(f, fmax)

		for psd in data[:]:
			axes.loglog(f[imin:imax], psd[imin:imax], alpha = 0.7)

		axes.grid(True)
		axes.set_xlim((fmin, fmax))
		axes.set_title(r"Spectral Density of %s, %s at %.9g s" % (self.instrument or "Unknown Instrument", self.channel_name or "Unknown Channel", float(inbuf.timestamp) / gst.SECOND))
		axes.set_xlabel(r"Frequency (Hz)")
		axes.set_ylabel(r"Spectral Density (FIXME)")

		#
		# copy pixel data to output buffer
		#

		matplotlibhelper.render(self.figure, outbuf)

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

		else:
			return super(Spectrum, self).do_transform_size(direction, caps, size, othercaps)


gobject.type_register(Spectrum)


def mkspectrumplot(pipeline, src, pad = None):
	elem = Spectrum()
	pipeline.add(elem)
	if pad is not None:
		src.link_pads(pad, elem, "sink")
	else:
		src.link(elem)
	return elem
