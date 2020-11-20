# Copyright (C) 2012 Madeline Wade
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

"""Produce LHO coherent and null data streams"""
__author__ = "Madeline Wade <madeline.wade@ligo.org>"

import scipy.fftpack
import numpy

import gobject
gobject.threads_init()
import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
import gst

from gstlal import pipeparts

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

class lal_lho_coherent_null(gst.Bin):

	__gstdetails__ = (
		'LHO Coherent and Null Streams',
		'Filter',
		__doc__,
		__author__
	)
	
	__gproperties__ = {
		'block-stride' : (
			gobject.TYPE_UINT,
			'block stride',
			'block stride for fir bank',
			1, gobject.G_MAXUINT, 1024,
			gobject.PARAM_READWRITE
		),
		'H1-impulse' : (
			gobject.TYPE_PYOBJECT,
			'H1 impulse',
			'impulse response for H1',
			gobject.PARAM_READWRITE
		),
		'H2-impulse' : (
			gobject.TYPE_PYOBJECT,
			'H2 impulse',
			'impulse response for H2',
			gobject.PARAM_READWRITE
		),
		'H1-latency' : (
			gobject.TYPE_UINT,
			'H1 latency',
			'latency for H1',
			0, gobject.G_MAXUINT, 0,
			gobject.PARAM_READWRITE
		),
		'H2-latency' : (
			gobject.TYPE_UINT,
			'H2 latency',
			'latency for H2',
			0, gobject.G_MAXUINT, 0,
			gobject.PARAM_READWRITE
		)
	}

	def do_set_property(self, prop, val):
		if prop.name == "block-stride":
			self.H1firfilter.set_property("block-stride", val)
			self.H2firfilter.set_property("block-stride", val)
		elif prop.name == "H1-impulse":
			self.H1firfilter.set_property("fir-matrix", [val])
		elif prop.name == "H2-impulse":
			self.H2firfilter.set_property("fir-matrix", [val])
		elif prop.name == "H1-latency":
			self.H1firfilter.set_property("latency", val)
		elif prop.name == "H2-latency":
			self.H2firfilter.set_property("latency", val)
		else:
			raise AssertionError

	def do_get_property(self, prop):
		if prop.name == "block-stride":
			return self.H1firfilter.get_property("block-stride")
		elif prop.name == "H1-impulse":
			return self.H1firfilter.get_property("fir-matrix")[0]
		elif prop.name == "H2-impulse":
			return self.H2firfilter.get_property("fir-matrix")[0]
		elif prop.name == "H1-latency":
			return self.H1firfilter.get_property("latency")
		elif prop.name == "H2-latency":
			return self.H2firfilter.get_property("latency")
		else:
			raise AssertionError

	def __init__(self):
		super(lal_lho_coherent_null, self).__init__()

		# tee off sources
		H1tee = gst.element_factory_make("tee")
		self.add(H1tee)
		H2tee = gst.element_factory_make("tee")
		self.add(H2tee)

		self.add_pad(gst.GhostPad("H1sink", H1tee.get_pad("sink")))
		self.add_pad(gst.GhostPad("H2sink", H2tee.get_pad("sink")))

		# apply fir filter to H1 data
		self.H1firfilter = H1head = pipeparts.mkfirbank(self, H1tee)

		# apply fir filter to H2 data
		self.H2firfilter = H2head = pipeparts.mkfirbank(self, H2tee)

		#
		# create coherent stream
		#

		COHhead = gst.element_factory_make("lal_adder")
		COHhead.set_property("sync", True)
		self.add(COHhead)
		pipeparts.mkqueue(self, H1head).link(COHhead)
		pipeparts.mkqueue(self, H2head).link(COHhead)

		#
		# create null stream
		#

		NULLhead = gst.element_factory_make("lal_adder")
		NULLhead.set_property("sync", True)
		self.add(NULLhead)
		pipeparts.mkqueue(self, H1tee).link(NULLhead)
		pipeparts.mkaudioamplify(self, pipeparts.mkqueue(self, H2tee), -1).link(NULLhead)

		self.add_pad(gst.GhostPad("COHsrc", COHhead.get_pad("src")))
		self.add_pad(gst.GhostPad("NULLsrc", NULLhead.get_pad("src")))

gobject.type_register(lal_lho_coherent_null)

__gstelementfactory__ = (
	lal_lho_coherent_null.__name__,
	gst.RANK_NONE,
	lal_lho_coherent_null
)
