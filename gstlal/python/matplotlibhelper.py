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


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import GObject, Gst, GstBase
GObject.threads_init()
Gst.init(None)

from gstlal.pipeutil import *
from gstlal import pipeio


"""Pad template suitable for producing video frames using Matplotlib.
The Agg backend supports rgba, argb, and bgra."""
padtemplate = Gst.PadTemplate.new(
	"src",
	Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS,
	Gst.caps_from_string("""
		video/x-raw,
		format = (string) {RGB, ARGB, RGBA, BGRA},
		width = (int) [1, MAX],
		height = (int) [1, MAX],
		framerate = (fraction) [0/1, MAX]
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


def render(fig, buf, dims, fmt):
	"""Render a Matplotlib figure to a GStreamer buffer."""
	width, height = dims
	fig.set_size_inches(
		width / float(fig.get_dpi()),
		height / float(fig.get_dpi())
	)
	fig.canvas.draw()
	if fmt == "RGB":
		imgdata = fig.canvas.renderer._renderer.tostring_rgb()
	elif fmt == "ARGB":
		imgdata = fig.canvas.renderer._renderer.tostring_argb()
	elif fmt == "RGBA":
		imgdata = fig.canvas.renderer._renderer.buffer_rgba()
	elif fmt == "BGRA":
		imgdata = fig.canvas.renderer._renderer.tostring_bgra()
	else:
		raise ValueError('invalid format "%s"' % fmt)
	datasize = len(imgdata)
	buf[:datasize] = imgdata
	buf.datasize = datasize


class BaseMatplotlibTransform(GstBase.BaseTransform):
	"""Base class for transform elements that use Matplotlib to render video."""

	__gsttemplates__ = padtemplate

	def __init__(self):
		self.figure = figure()
		self.axes = self.figure.gca()

	def do_transform_caps(self, direction, caps):
		"""GstBaseTransform->transform_caps virtual method."""
		if direction == Gst.PadDirection.SRC:
			return self.get_static_pad("sink").get_fixed_caps_func()
		elif direction == Gst.PadDirection.SINK:
			return self.get_static_pad("src").get_fixed_caps_func()
		raise ValueError(direction)

	def do_transform_size(self, direction, caps, size, othercaps):
		"""GstBaseTransform->transform_size virtual method."""
		if direction == Gst.PadDirection.SINK:
			return pipeio.get_unit_size(self.get_static_pad("src").query_caps(None))
		raise ValueError(direction)

GObject.type_register(BaseMatplotlibTransform) # MOD: Found type_register in line: [gobject.type_register(BaseMatplotlibTransform)]
