# Copyright (C) 2010 Erin Kara
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
Inputs data from lal_skymap element from 4 channels of double precision values
to out a skymap plot usins matplotlib

Source:
channel 0:  theta: geographical polar elevation coordinate, theta (radians)
channel 1:    phi: geographical polar azimuthal coordinate, phi (radians)
channel 2:   span: length of point (radian)
channel 3: log(P): probability associated with point
"""
__author__ = "Erin Kara <erin.kara@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import sys
from gstlal.pipeutil import *

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
from matplotlib import cm
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy

from gstlal import pipeio
from gstlal.elements import matplotlibcaps
from gst.extend.pygobject import gproperty, with_construct_properties


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#



class lal_skymaprenderer(gst.BaseTransform):
	__gstdetails__ = (
		'Skymap Plotting Element',
		'Transform',
		__doc__,
		__author__
	)

	gproperty(
		gobject.TYPE_BOOLEAN,
		"entire-sky",
		"Set to TRUE to view a grid spanning the entire sky",
		True,
		construct=True
	)

	gproperty(
		gobject.TYPE_BOOLEAN,
		"colormap",
		'Set to TRUE for colored map, FALSE for grayscale)',
		True,
		construct=True
	)

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"channels = (int) 4, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				matplotlibcaps + ", " +
				"width = (int) [1, MAX], " +
				"height = (int) [1, MAX]"
			)
		)
	)


	def __init__(self):
		super(lal_skymaprenderer, self).__init__()
		self.out_height = 300   # default, pixels
		self.out_width = 600    # default, pixels
		self.set_property("entire-sky", True)
		self.set_property("colormap", True)

	def do_get_unit_size(self, caps):
		return pipeio.get_unit_size(caps)

	def do_set_caps(self, incaps, outcaps):
		channels = incaps[0]["channels"]
		self.channels = channels
		self.out_width = outcaps[0]["width"]
		self.out_height = outcaps[0]["height"]
		return True


	def do_transform(self, inbuf, outbuf):
		#
		# Render Skymap from list of tuples with geographic latitude,
		# geographic longitude, span, log(probability)
		#

		# Set buffer metadata
		outbuf.offset = inbuf.offset
		outbuf.offset_end = inbuf.offset_end
		outbuf.duration = inbuf.duration
		outbuf.timestamp = inbuf.timestamp


		# Convert raw buffer into numpy array
		skymap_array = pipeio.array_from_audio_buffer(inbuf)



		# Assign column of skymap_array as individual arrays
		theta = skymap_array[:,0]
		phi = skymap_array[:,1]
		span = skymap_array[:,2]
		logp = skymap_array[:,3]



		# Generate plot
		fig = figure.Figure()
		FigureCanvas(fig)
		fig.set_size_inches(self.out_width / float(fig.get_dpi()), self.out_height / float(fig.get_dpi()))
		axes = fig.gca()
		cax, kw = matplotlib.colorbar.make_axes(axes)

		# Specify vertices of polygons
		phi_vertices = numpy.array([phi-span/2, phi-span/2, phi+span/2, phi+span/2])
		phi_vertices = phi_vertices.T
		theta_vertices = numpy.array([theta-span/2, theta+span/2, theta+span/2, theta-span/2])
		theta_vertices = theta_vertices.T

		# Allow for jet or gray colormap
		if (self.get_property("colormap")):
			colormap = matplotlib.colorbar.ColorbarBase(cax, norm=colors.Normalize(vmin=logp.min(), vmax=logp.max()), cmap=cm.jet)
		else:
			colormap = matplotlib.colorbar.ColorbarBase(cax, norm=colors.Normalize(vmin=logp.min(), vmax=logp.max()), cmap=cm.gray)

		# Fill polygons with logp
		for i in range(len(logp)):
			axes.fill(phi_vertices[i], theta_vertices[i], facecolor=colormap.to_rgba(logp[i]), edgecolor='none')
		axes.set_title(r"Signal Candidate Probability Distribution")
		axes.set_xlabel(r"Geographic Latitude (radians)")
		axes.set_ylabel(r"Geographic Longitude (radians)")



		if self.get_property("entire-sky"):
			axes.set_xlim((0, 2*numpy.pi))
			axes.set_ylim((0, numpy.pi))
			axes.grid(True)
			axes.invert_yaxis()



		# Extract pixel data
		fig.canvas.draw()
		rgba_buffer = fig.canvas.buffer_rgba(0,0)
		rgba_buffer_size = len(rgba_buffer)



		# Copy pixel data
		outbuf[0:rgba_buffer_size] = rgba_buffer
		outbuf.datasize = rgba_buffer_size



		# Done
		return gst.FLOW_OK



	def do_transform_caps(self, direction, caps):
		if direction == gst.PAD_SRC:
			#
			# convert src pad's caps to sink pad's
			#

			return self.get_pad("sink").get_fixed_caps_func()

		elif direction == gst.PAD_SINK:
			#
			# convert sink pad's caps to src pad's
			#

			return self.get_pad("src").get_fixed_caps_func()

		raise ValueError

	def do_transform_size(self, direction, caps, size, othercaps):


		if direction == gst.PAD_SINK:


			# FIXME:  why is othercaps not the *other* caps?
			return self.out_width * self.out_height * 4
                        return othercaps[0]["width"] * othercaps[0]["height"] * othercaps[0]["bpp"] / 8

		raise ValueError, direction

# Register element class
gstlal_element_register(lal_skymaprenderer)
