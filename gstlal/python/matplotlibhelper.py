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

## @file

## @package matplotlibhelper
"""
Classes and functions for building Matplotlib-based GStreamer elements
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"
__all__ = ("padtemplate", "figure", "render", "BaseMatplotlibTransform")


from gstlal.pipeutil import *
from gstlal import pipeio


"""Pad template suitable for producing video frames using Matplotlib.
The Agg backend supports rgba, argb, and bgra."""
padtemplate = gst.PadTemplate(
	"src",
	gst.PAD_SRC, gst.PAD_ALWAYS,
	gst.caps_from_string("""
		video/x-raw-rgb,
		bpp        = (int) {24,32},
		depth      = (int) 24,
		endianness = (int) BIG_ENDIAN,
		red_mask   = (int) 0xFF0000,
		green_mask = (int) 0x00FF00,
		blue_mask  = (int) 0x0000FF;
		video/x-raw-rgb,
		bpp        = (int) 32,
		depth      = (int) {24,32},
		endianness = (int) BIG_ENDIAN,
		red_mask   = (int) 0x00FF0000,
		green_mask = (int) 0x0000FF00,
		blue_mask  = (int) 0x000000FF,
		alpha_mask = (int) 0xFF000000;
		video/x-raw-rgb,
		bpp        = (int) 32,
		depth      = (int) {24,32},
		endianness = (int) BIG_ENDIAN,
		red_mask   = (int) 0x0000FF00,
		green_mask = (int) 0x00FF0000,
		blue_mask  = (int) 0xFF000000,
		alpha_mask = (int) 0x000000FF;
	""")
)


def figure():
	"""Create a Matplotlib Figure object suitable for rendering video frames."""
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
		"text.usetex": False,
		"path.simplify": True
	})
	from matplotlib import figure
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	figure = figure.Figure()
	FigureCanvasAgg(figure)
	return figure


def render(fig, buf):
	"""Render a Matplotlib figure to a GStreamer buffer."""
	caps = buf.caps[0]
	fig.set_size_inches(
		caps['width'] / float(fig.get_dpi()),
		caps['height'] / float(fig.get_dpi())
	)
	fig.canvas.draw()
	if caps['bpp'] == 24: # RGB
		imgdata = fig.canvas.renderer._renderer.tostring_rgb()
	elif caps['alpha_mask'] & 0xFF000000 == 0xFF000000: # ARGB
		imgdata = fig.canvas.renderer._renderer.tostring_argb()
	elif caps['red_mask'] == 0xFF: # RGBA
		imgdata = fig.canvas.renderer._renderer.buffer_rgba()
	else: # BGRA
		imgdata = fig.canvas.renderer._renderer.tostring_bgra()
	datasize = len(imgdata)
	buf[:datasize] = imgdata
	buf.datasize = datasize


class BaseMatplotlibTransform(gst.BaseTransform):
	"""Base class for transform elements that use Matplotlib to render video."""

	__gsttemplates__ = padtemplate

	def __init__(self):
		self.figure = figure()
		self.axes = self.figure.gca()

	def do_transform_caps(self, direction, caps):
		"""GstBaseTransform->transform_caps virtual method."""
		if direction == gst.PAD_SRC:
			return self.get_pad("sink").get_fixed_caps_func()
		elif direction == gst.PAD_SINK:
			return self.get_pad("src").get_fixed_caps_func()
		raise ValueError(direction)

	def do_transform_size(self, direction, caps, size, othercaps):
		"""GstBaseTransform->transform_size virtual method."""
		if direction == gst.PAD_SINK:
			return pipeio.get_unit_size(self.get_pad("src").get_caps())
		raise ValueError(direction)

gobject.type_register(BaseMatplotlibTransform)
