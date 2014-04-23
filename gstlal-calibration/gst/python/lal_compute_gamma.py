# Copyright (C) 2014 Madeline Wade
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

"""Compute the time dependent gain of the sensing function, known as \gamma(t), for aLIGO calibration"""
__author__ = "Madeline Wade <madeline.wade@ligo.org>"

import gobject
gobject.threads_init()
import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
import gst

from gstlal import pipeparts
import numpy

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

class lal_compute_gamma(gst.Bin):
	# Set default values for properties
	olgR_default = 1.0
	olgI_default = 1.0
	wR_default = 1.0
	wI_default = 1.0
	#cal_line_freq_default = 1144.3
	sr_default = 16384
	time_domain_default = True
	#caps_default = 'audio/x-raw-float, width=64, rate=16384'

	__gstdetails__ = (
		'Compute Gamma',
		'Filter',
		__doc__,
		__author__
	)

	__gproperties__ = {
		'olgR' : (
			gobject.TYPE_DOUBLE,
			'olgR',
			'real part of open loop gain at calibration line frequency',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, olgR_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'olgI' : (
			gobject.TYPE_DOUBLE,
			'olgI',
			'imaginary part of open loop gain at calibration line frequency',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, olgI_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'wR' : (
			gobject.TYPE_DOUBLE,
			'wR',
			'real part of darm_ctrl whitening filter at calibration line frequency',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, wR_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'wI' : (
			gobject.TYPE_DOUBLE,
			'wI',
			'imaginary part of darm_ctrl whitening filter at calibration line frequency',
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, wI_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
#		'cal-line-freq' : (
#			gobject.TYPE_DOUBLE,
#			'cal line freq',
#			'calibration line frequency',
#			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, cal_line_freq_default,
#			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
#		),
		'sr' : (
			gobject.TYPE_UINT,
			'sr',
			'sample rate',
			0, gobject.G_MAXUINT, sr_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'time-domain' : (
			gobject.TYPE_BOOLEAN,
			'time domain',
			'set to True to perform FIR filtering in the time domain',
			time_domain_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
#		'caps' : (
#			gobject.TYPE_STRING,
#			'caps',
#			'caps for each data stream',
#			caps_default,
#			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
#		)
	}

	def do_set_property(self, prop, val):
#		if prop.name == 'cal-line-freq':
#			self.cal_line_freq = val
		if prop.name == 'sr':
			self.sr = val
#		elif prop.name == 'caps':
#			self.caps = val
		elif prop.name == 'time-domain':
			self.time_domain = val
		elif prop.name == 'olgR':
			self.olgR = val
		elif prop.name == 'olgI':
			self.olgI = val
		elif prop.name == 'wR':
			self.wR = val
		elif prop.name == 'wI':
			self.wI = val

	def do_get_property(self, prop):
#		if prop.name == 'cal-line-freq':
#			return self.cal_line_freq
		if prop.name == 'sr':
			return self.sr
#		elif prop.name == 'caps':
#			return self.caps
		elif prop.name == 'time-domain':
			return self.time_domain
		elif prop.name == 'olgR':
			return self.olgR
		elif prop.name == 'olgI':
			return self.olgI
		elif prop.name == 'wR':
			return self.wR
		elif prop.name == 'wI':
			return self.wI


	def __init__(self):
		super(lal_compute_gamma, self).__init__()
		
		#self.cal_line_freq, self.sr, self.caps, self.time_domain, self.olgR, self.olgI, self.wR, self.wI = lal_compute_gamma.cal_line_freq_default, lal_compute_gamma.sr_default, lal_compute_gamma.caps_default, lal_compute_gamma.time_domain_default, lal_compute_gamma.olgR_default, lal_compute_gamma.olgI_default, lal_compute_gamma.wR_default, lal_compute_gamma.wI_default
		self.sr, self.time_domain, self.olgR, self.olgI, self.wR, self.wI = lal_compute_gamma.sr_default, lal_compute_gamma.time_domain_default, lal_compute_gamma.olgR_default, lal_compute_gamma.olgI_default, lal_compute_gamma.wR_default, lal_compute_gamma.wI_default

		self.w_mod = self.wR*self.wR + self.wI*self.wI
		self.olg_mod = self.olgR*self.olgR + self.olgI*self.olgI
#		self.deltat = 1.0/self.sr
		self.window = numpy.hanning(self.sr)

		# Make an oscillator at calibration line frequency for computation of gamma factors
		#re_osc = pipeparts.mkgeneric(self, None, "lal_numpy_functiongenerator", expression = "%f * cos(2.0 * 3.1415926535897931 * %f * t)" % (self.deltat, self.cal_line_freq), blocksize = self.sr * 4)
		#re_osc = pipeparts.mkcapsfilter(self, re_osc, self.caps)
		cos = gst.element_factory_make("tee")
		self.add(cos)
		sin = gst.element_factory_make("tee")
		self.add(sin)
		
		self.add_pad(gst.GhostPad("cos", cos.get_pad("sink")))
		self.add_pad(gst.GhostPad("sin", sin.get_pad("sink")))		

		#im_osc = pipeparts.mkgeneric(self, None, "lal_numpy_functiongenerator", expression = "-1.0 * %f * sin(2.0 * 3.1415926535897931 * %f * t)" % (self.deltat, self.cal_line_freq), blocksize = self.sr * 4)
		#im_osc = pipeparts.mkcapsfilter(self, im_osc, self.caps)
		#im_osc = pipeparts.mktee(self, im_osc)

		# Tee off sources
		exc = gst.element_factory_make("tee")
		self.add(exc)
		dctrl = gst.element_factory_make("tee")
		self.add(dctrl)
		
		self.add_pad(gst.GhostPad("exc_sink", exc.get_pad("sink")))
		self.add_pad(gst.GhostPad("dctrl_sink", dctrl.get_pad("sink")))

		# EXC branch for gamma
		excR = gst.element_factory_make("lal_multiplier")
		excR.set_property("sync", True)
		self.add(excR)
		pipeparts.mkqueue(self, exc).link(excR)
		pipeparts.mkqueue(self, cos).link(excR)
		excR = pipeparts.mkfirbank(self, excR, fir_matrix = [self.window], time_domain = self.time_domain)
		excR = pipeparts.mktee(self, excR)

		excI = gst.element_factory_make("lal_multiplier")
		excI.set_property("sync", True)
		self.add(excI)
		pipeparts.mkqueue(self, exc).link(excI)
		pipeparts.mkqueue(self, sin).link(excI)
		excI = pipeparts.mkfirbank(self, excI, fir_matrix = [self.window], time_domain = self.time_domain)
		excI = pipeparts.mktee(self, excI)

		# DARM_CTRL branch for gamma
		dctrlR = gst.element_factory_make("lal_multiplier")
		dctrlR.set_property("sync", True)
		self.add(dctrlR)
		pipeparts.mkqueue(self, dctrl).link(dctrlR)
		pipeparts.mkqueue(self, cos).link(dctrlR)
		dctrlR = pipeparts.mkfirbank(self, dctrlR, fir_matrix = [self.window], time_domain = self.time_domain)
		dctrlR = pipeparts.mktee(self, dctrlR)

		dctrlI = gst.element_factory_make("lal_multiplier")
		dctrlI.set_property("sync", True)
		self.add(dctrlI)
		pipeparts.mkqueue(self, dctrl).link(dctrlI)
		pipeparts.mkqueue(self, sin).link(dctrlI)
		dctrlI = pipeparts.mkfirbank(self, dctrlI, fir_matrix = [self.window], time_domain = self.time_domain)
		dctrlI = pipeparts.mktee(self, dctrlI)

		# Make useful combos of channels for calculating gamma
		dctrl_mod = gst.element_factory_make("lal_adder")
		dctrl_mod.set_property("sync", True)
		self.add(dctrl_mod)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, dctrlR, exponent = 2.0)).link(dctrl_mod)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, dctrlI, exponent = 2.0)).link(dctrl_mod)
		dctrl_mod = pipeparts.mktee(self, dctrl_mod)

		dctrl_mod_w_mod_olgR = pipeparts.mkaudioamplify(self, dctrl_mod, -1.0 * self.olgR * self.w_mod)
		dctrl_mod_w_mod_olgI = pipeparts.mkaudioamplify(self, dctrl_mod, self.olgI * self.w_mod)
	
		dctrlR_excI = gst.element_factory_make("lal_multiplier")
		dctrlR_excI.set_property("sync", True)
		self.add(dctrlR_excI)
		pipeparts.mkqueue(self, dctrlR).link(dctrlR_excI)
		pipeparts.mkqueue(self, excI).link(dctrlR_excI)
		dctrlR_excI = pipeparts.mktee(self, dctrlR_excI)

		dctrlR_excI_olgI_wI = pipeparts.mkaudioamplify(self, dctrlR_excI, -1.0 * self.olgI * self.wI)
		dctrlR_excI_olgI_wR = pipeparts.mkaudioamplify(self, dctrlR_excI, self.olgI * self.wR)
		dctrlR_excI_olgR_wI = pipeparts.mkaudioamplify(self, dctrlR_excI, self.olgR * self.wI)
		dctrlR_excI_olgR_wR = pipeparts.mkaudioamplify(self, dctrlR_excI, self.olgR * self.wR)

		dctrlI_excR = gst.element_factory_make("lal_multiplier")
		dctrlI_excR.set_property("sync", True)
		self.add(dctrlI_excR)
		pipeparts.mkqueue(self, dctrlI).link(dctrlI_excR)
		pipeparts.mkqueue(self, excR).link(dctrlI_excR)
		dctrlI_excR = pipeparts.mktee(self, dctrlI_excR)

		dctrlI_excR_olgI_wR = pipeparts.mkaudioamplify(self, dctrlI_excR, -1.0 * self.olgI * self.wR)
		dctrlI_excR_olgI_wI = pipeparts.mkaudioamplify(self, dctrlI_excR, self.olgI * self.wI)
		dctrlI_excR_olgR_wI = pipeparts.mkaudioamplify(self, dctrlI_excR, -1.0 * self.olgR * self.wI)
		dctrlI_excR_olgR_wR = pipeparts.mkaudioamplify(self, dctrlI_excR, -1.0 * self.olgR * self.wR)

		dctrlI_excI = gst.element_factory_make("lal_multiplier")
		dctrlI_excI.set_property("sync", True)
		self.add(dctrlI_excI)
		pipeparts.mkqueue(self, dctrlI).link(dctrlI_excI)
		pipeparts.mkqueue(self, excI).link(dctrlI_excI)
		dctrlI_excI = pipeparts.mktee(self, dctrlI_excI)

		dctrlI_excI_olgR_wR = pipeparts.mkaudioamplify(self, dctrlI_excI, self.olgR * self.wR)
		dctrlI_excI_olgR_wI = pipeparts.mkaudioamplify(self, dctrlI_excI, -1.0 * self.olgR * self.wI)
		dctrlI_excI_olgI_wI = pipeparts.mkaudioamplify(self, dctrlI_excI, -1.0 * self.olgI * self.wI)
		dctrlI_excI_olgI_wR = pipeparts.mkaudioamplify(self, dctrlI_excI, -1.0 * self.olgI * self.wR)

		dctrlR_excR = gst.element_factory_make("lal_multiplier")
		dctrlR_excR.set_property("sync", True)
		self.add(dctrlR_excR)
		pipeparts.mkqueue(self, dctrlR).link(dctrlR_excR)
		pipeparts.mkqueue(self, excR).link(dctrlR_excR)
		dctrlR_excR = pipeparts.mktee(self, dctrlR_excR)

		dctrlR_excR_olgR_wR = pipeparts.mkaudioamplify(self, dctrlR_excR, self.olgR * self.wR)
		dctrlR_excR_olgR_wI = pipeparts.mkaudioamplify(self, dctrlR_excR, -1.0 * self.olgR * self.wI)
		dctrlR_excR_olgI_wI = pipeparts.mkaudioamplify(self, dctrlR_excR, -1.0 * self.olgI * self.wI)
		dctrlR_excR_olgI_wR = pipeparts.mkaudioamplify(self, dctrlR_excR, -1.0 * self.olgI * self.wR)

		# Combine all of these branches into real and imaginary gamma
		denom = pipeparts.mkaudioamplify(self, dctrl_mod, self.olg_mod * self.w_mod)
		denom = pipeparts.mktee(self, denom)

		gammaI_num = gst.element_factory_make("lal_adder")
		gammaI_num.set_property("sync", True)
		self.add(gammaI_num)
		pipeparts.mkqueue(self, dctrl_mod_w_mod_olgI).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgI_wI).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlI_excR_olgI_wI).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlI_excI_olgR_wI).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlR_excR_olgR_wI).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlI_excI_olgI_wR).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlR_excR_olgI_wR).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgR_wR).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlI_excR_olgR_wR).link(gammaI_num)

		gammaR_num = gst.element_factory_make("lal_adder")
		gammaR_num.set_property("sync", True)
		self.add(gammaR_num)
		pipeparts.mkqueue(self, dctrl_mod_w_mod_olgR).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlI_excI_olgI_wI).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excR_olgI_wI).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgR_wI).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlI_excR_olgR_wI).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgI_wR).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlI_excR_olgI_wR).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlI_excI_olgR_wR).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excR_olgR_wR).link(gammaR_num)

		gammaR = gst.element_factory_make("lal_multiplier")
		gammaR.set_property("sync", True)
		self.add(gammaR)
		pipeparts.mkqueue(self, gammaR_num).link(gammaR)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, denom, exponent = -1.0)).link(gammaR)

		gammaI = gst.element_factory_make("lal_multiplier")
		gammaI.set_property("sync", True)
		self.add(gammaI)
		pipeparts.mkqueue(self, gammaI_num).link(gammaI)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, denom, exponent = -1.0)).link(gammaI)

		# Set up src pads
		self.add_pad(gst.GhostPad("gammaR", gammaR.get_pad("src")))
		self.add_pad(gst.GhostPad("gammaI", gammaI.get_pad("src")))

gobject.type_register(lal_compute_gamma)

__gstelementfactory__ = (
	lal_compute_gamma.__name__,
	gst.RANK_NONE,
	lal_compute_gamma
)
