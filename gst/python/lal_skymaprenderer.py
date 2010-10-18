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

import numpy

from gstlal.pipeutil import *
from gstlal import pipeio
from gstlal import matplotlibhelper

import matplotlib
from matplotlib import cm
from matplotlib import colors


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


class lal_skymaprenderer(matplotlibhelper.BaseMatplotlibTransform):
	__gstdetails__ = (
		'Skymap Plotting Element',
		'Transform',
		__doc__,
		__author__
	)


	__gsttemplates__ = (
		matplotlibhelper.BaseMatplotlibTransform.__gsttemplates__,
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"channels = (int) 4, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		)
	)

	def __init__(self):
		super(lal_skymaprenderer, self).__init__() 
		self.cax, kw = matplotlib.colorbar.make_axes(self.axes)



	def do_transform(self, inbuf, outbuf):
		#
		# Render Skymap from list of tuples with geographic latitude,
		# geographic longitude, span, log(probability)
		#


		# Convert raw buffer into numpy array
		skymap_array = pipeio.array_from_audio_buffer(inbuf)


		# Assign column of skymap_array as individual arrays
		theta = skymap_array[:,0]
		phi = skymap_array[:,1]
		span = skymap_array[:,2]
		logp = skymap_array[:,3]


		# Clear old axes
		self.cax.cla()
		self.axes.cla()

		# Specify vertices of polygons
		phi_vertices = numpy.array([phi-span/2, phi-span/2, phi+span/2, phi+span/2])
		phi_vertices = phi_vertices.T
		theta_vertices = numpy.array([theta-span/2, theta+span/2, theta+span/2, theta-span/2])
		theta_vertices = theta_vertices.T


		# Make ColorbarBase object
		colormap = matplotlib.colorbar.ColorbarBase(self.cax, norm=colors.Normalize(vmin=logp.min(), vmax=logp.max()), cmap=cm.jet)


		# Fill polygons with logp
		for i in range(len(logp)):
			self.axes.fill(phi_vertices[i], theta_vertices[i], facecolor=colormap.to_rgba(logp[i]), edgecolor='none')
		self.axes.set_title(r"Signal Candidate Probability Distribution")
		self.axes.set_xlabel(r"Geographic Latitude (radians)")
		self.axes.set_ylabel(r"Geographic Longitude (radians)")


		# Render to output buffer
		matplotlibhelper.render(self.figure, outbuf)


		# Set buffer metadata
		outbuf.offset = inbuf.offset
		outbuf.offset_end = inbuf.offset_end
		outbuf.duration = inbuf.duration
		outbuf.timestamp = inbuf.timestamp		


		# Done
		return gst.FLOW_OK



# Register element class
gstlal_element_register(lal_skymaprenderer)
