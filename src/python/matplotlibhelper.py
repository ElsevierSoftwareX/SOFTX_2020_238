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
Classes and functions for building Matplotlib-based GStreamer elements
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import gst


"""Pad template suitable for producing video frames using Matplotlib."""
padtemplate = gst.PadTemplate(
	"src",
	gst.PAD_SRC, gst.PAD_ALWAYS,
	gst.caps_from_string("""
		video/x-raw-rgb,
		bpp        = (int) 32,
		depth      = (int) 32,
		endianness = (int) BIG_ENDIAN,
		red_mask   = (int) 0x00FF0000,
		green_mask = (int) 0x0000FF00,
		blue_mask  = (int) 0x000000FF,
		alpha_mask = (int) 0xFF000000;
		video/x-raw-rgb,
		bpp        = (int) 24,
		depth      = (int) 24,
		endianness = (int) BIG_ENDIAN,
		red_mask   = (int) 0xFF0000,
		green_mask = (int) 0x00FF00,
		blue_mask  = (int) 0x0000FF
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
	fig.set_size_inches(
		buf.caps[0]['width'] / float(fig.get_dpi()),
		buf.caps[0]['height'] / float(fig.get_dpi())
	)
	fig.canvas.draw()
	if buf.caps[0]['depth'] == 24:
		img_str = fig.canvas.tostring_rgb()
	else:
		img_str = fig.canvas.tostring_argb()
	datasize = len(img_str)
	buf[:datasize] = img_str
	buf.datasize = datasize
