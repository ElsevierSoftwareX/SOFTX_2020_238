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
	wmod_default = 1.0
	olgmod_default = 1.0
	sr_default = 16384
	time_domain_default = True
	integration_samples_default = sr_default

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
		'wmod' : (
			gobject.TYPE_DOUBLE,
			'wmod',
			'modulus of darm_ctrl whitening filter at calibration line frequency',
			0.0, gobject.G_MAXDOUBLE, wmod_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'olgmod' : (
			gobject.TYPE_DOUBLE,
			'olgmod',
			'modulus of open loop gain at calibration line frequency',
			0.0, gobject.G_MAXDOUBLE, olgmod_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT,
		),
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
		),
		'integration-samples' : (
			gobject.TYPE_UINT,
			'integration time',
			'number of samples in integration',
			0, gobject.G_MAXUINT, integration_samples_default,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}

	def do_set_property(self, prop, val):
		if prop.name == 'sr':
			self.sr = val
			self.excR_capsfilter.set_property("caps", gst.Caps("audio/x-raw-float, rate=%d" % val))
			self.excI_capsfilter.set_property("caps", gst.Caps("audio/x-raw-float, rate=%d" % val))
			self.dctrlR_capsfilter.set_property("caps", gst.Caps("audio/x-raw-float, rate=%d" % val))
			self.dctrlI_capsfilter.set_property("caps", gst.Caps("audio/x-raw-float, rate=%d" % val))
		elif prop.name == 'time-domain':
			self.time_domain = val
			self.excR_firbank.set_property("time-domain", val)
			self.excI_firbank.set_property("time-domain", val)
			self.dctrlR_firbank.set_property("time-domain", val)
			self.dctrlI_firbank.set_property("time-domain", val)
		elif prop.name == 'integration-samples': 
			self.integration_samples = val
			self.excR_firbank.set_property("fir-matrix", [numpy.hanning(val+1)])
			self.excI_firbank.set_property("fir-matrix", [numpy.hanning(val+1)])
			self.dctrlR_firbank.set_property("fir-matrix", [numpy.hanning(val+1)])
			self.dctrlI_firbank.set_property("fir-matrix", [numpy.hanning(val+1)])
		elif prop.name == 'olgR':
			self.olgR = val
			self.dctrl_mod_w_mod_olgR.set_property("amplification", val)
			self.dctrlR_excI_olgR.set_property("amplification", val)
			self.dctrlI_excR_olgR.set_property("amplification", val)
			self.dctrlI_excI_olgR.set_property("amplification", val)
			self.dctrlR_excR_olgR.set_property("amplification", val)
		elif prop.name == 'olgI':
			self.olgI = val
			self.dctrl_mod_w_mod_olgI.set_property("amplification", val)
			self.dctrlR_excI_olgI.set_property("amplification", val)
			self.dctrlI_excR_olgI.set_property("amplification", val)
			self.dctrlI_excI_olgI.set_property("amplification", val)
			self.dctrlR_excR_olgI.set_property("amplification", val)
		elif prop.name == 'wR':
			self.wR = val
			self.dctrlR_excI_olgI_wR.set_property("amplification", val)
			self.dctrlR_excI_olgR_wR.set_property("amplification", val)
			self.dctrlI_excR_olgI_wR.set_property("amplification", val)
			self.dctrlI_excR_olgR_wR.set_property("amplification", val)
			self.dctrlI_excI_olgR_wR.set_property("amplification", val)
			self.dctrlI_excI_olgI_wR.set_property("amplification", val)
			self.dctrlR_excR_olgR_wR.set_property("amplification", val)
			self.dctrlR_excR_olgI_wR.set_property("amplification", val)
		elif prop.name == 'wI':
			self.wI = val
			self.dctrlR_excI_olgI_wI.set_property("amplification", val)
			self.dctrlR_excI_olgR_wI.set_property("amplification", val)
			self.dctrlI_excR_olgI_wI.set_property("amplification", val)
			self.dctrlI_excR_olgR_wI.set_property("amplification", val)
			self.dctrlI_excI_olgR_wI.set_property("amplification", val)
			self.dctrlI_excI_olgI_wI.set_property("amplification", val)
			self.dctrlR_excR_olgR_wI.set_property("amplification", val)
			self.dctrlR_excR_olgI_wI.set_property("amplification", val)
		elif prop.name == 'wmod':
			self.wmod = val
			self.dctrl_mod_w_mod.set_property("amplification", val)
		elif prop.name == 'olgmod':
			self.olgmod = val
			self.dctrl_mod_w_mod_olg_mod.set_property("amplification", val)
		else:
			raise AssertionError

	def do_get_property(self, prop):
		if prop.name == 'sr':
			return self.sr
		elif prop.name == 'time-domain':
			return self.time_domain
		elif prop.name == 'integration-samples':
			return self.integration_samples
		elif prop.name == 'olgR':
			return self.olgR
		elif prop.name == 'olgI':
			return self.olgI
		elif prop.name == 'wR':
			return self.wR
		elif prop.name == 'wI':
			return self.wI
		elif prop.name == 'wmod':
			return self.wmod
		elif prop.name == 'olgmod':
			return self.olgmod
		else:
			raise AssertionError


	def __init__(self):
		super(lal_compute_gamma, self).__init__()
		
		# Make an oscillator at calibration line frequency for computation of gamma factors
		cos = gst.element_factory_make("tee")
		self.add(cos)
		sin = gst.element_factory_make("tee")
		self.add(sin)
		
		self.add_pad(gst.GhostPad("cos", cos.get_pad("sink")))
		self.add_pad(gst.GhostPad("sin", sin.get_pad("sink")))		
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
		#excR = pipeparts.mkresample(self, excR, quality=9)
		excR = pipeparts.mkaudioundersample(self, excR)
		self.excR_capsfilter = excR = pipeparts.mkgeneric(self, excR, "capsfilter")
		self.excR_firbank = excR = pipeparts.mkfirbank(self, excR)	
		excR = pipeparts.mktee(self, excR)

		excI = gst.element_factory_make("lal_multiplier")
		excI.set_property("sync", True)
		self.add(excI)
		pipeparts.mkqueue(self, exc).link(excI)
		pipeparts.mkqueue(self, sin).link(excI)
		#excI = pipeparts.mkresample(self, excI, quality=9)
		excI = pipeparts.mkaudioundersample(self, excI)
		self.excI_capsfilter = excI = pipeparts.mkgeneric(self, excI, "capsfilter")
		self.excI_firbank = excI = pipeparts.mkfirbank(self, excI)
		excI = pipeparts.mktee(self, excI)

		# DARM_CTRL branch for gamma
		dctrlR = gst.element_factory_make("lal_multiplier")
		dctrlR.set_property("sync", True)
		self.add(dctrlR)
		pipeparts.mkqueue(self, dctrl).link(dctrlR)
		pipeparts.mkqueue(self, cos).link(dctrlR)
		dctrlR = pipeparts.mkresample(self, dctrlR, quality=9)
		self.dctrlR_capsfilter = dctrlR = pipeparts.mkgeneric(self, dctrlR, "capsfilter")
		self.dctrlR_firbank = dctrlR = pipeparts.mkfirbank(self, dctrlR)
		dctrlR = pipeparts.mktee(self, dctrlR)

		dctrlI = gst.element_factory_make("lal_multiplier")
		dctrlI.set_property("sync", True)
		self.add(dctrlI)
		pipeparts.mkqueue(self, dctrl).link(dctrlI)
		pipeparts.mkqueue(self, sin).link(dctrlI)
		dctrlI = pipeparts.mkresample(self, dctrlI, quality=9)
		self.dctrlI_capsfilter = dctrlI = pipeparts.mkgeneric(self, dctrlI, "capsfilter")
		self.dctrlI_firbank = dctrlI = pipeparts.mkfirbank(self, dctrlI)
		dctrlI = pipeparts.mktee(self, dctrlI)

		# Make useful combos of channels for calculating gamma
		dctrl_mod = gst.element_factory_make("lal_adder")
		dctrl_mod.set_property("sync", True)
		self.add(dctrl_mod)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, dctrlR, exponent = 2.0)).link(dctrl_mod)
		pipeparts.mkqueue(self, pipeparts.mkpow(self, dctrlI, exponent = 2.0)).link(dctrl_mod)

		self.dctrl_mod_w_mod = dctrl_mod_w_mod = pipeparts.mkgeneric(self, dctrl_mod, "audioamplify")
		dctrl_mod_w_mod = pipeparts.mktee(self, dctrl_mod_w_mod)
		self.dctrl_mod_w_mod_olgR = dctrl_mod_w_mod_olgR = pipeparts.mkgeneric(self, dctrl_mod_w_mod, "audioamplify")
		self.dctrl_mod_w_mod_olgI = dctrl_mod_w_mod_olgI = pipeparts.mkgeneric(self, dctrl_mod_w_mod, "audioamplify")
		self.dctrl_mod_w_mod_olg_mod = dctrl_mod_w_mod_olg_mod = pipeparts.mkgeneric(self, dctrl_mod_w_mod, "audioamplify")
	
		dctrlR_excI = gst.element_factory_make("lal_multiplier")
		dctrlR_excI.set_property("sync", True)
		self.add(dctrlR_excI)
		pipeparts.mkqueue(self, dctrlR).link(dctrlR_excI)
		pipeparts.mkqueue(self, excI).link(dctrlR_excI)
		dctrlR_excI = pipeparts.mktee(self, dctrlR_excI)

		self.dctrlR_excI_olgI = dctrlR_excI_olgI = pipeparts.mkgeneric(self, dctrlR_excI, "audioamplify")
		dctrlR_excI_olgI = pipeparts.mktee(self, dctrlR_excI_olgI)
		self.dctrlR_excI_olgI_wI = dctrlR_excI_olgI_wI = pipeparts.mkgeneric(self, dctrlR_excI_olgI, "audioamplify")
		self.dctrlR_excI_olgI_wR = dctrlR_excI_olgI_wR = pipeparts.mkgeneric(self, dctrlR_excI_olgI, "audioamplify")

		self.dctrlR_excI_olgR = dctrlR_excI_olgR = pipeparts.mkgeneric(self, dctrlR_excI, "audioamplify")
		dctrlR_excI_olgR = pipeparts.mktee(self, dctrlR_excI_olgR)
		self.dctrlR_excI_olgR_wI = dctrlR_excI_olgR_wI = pipeparts.mkgeneric(self, dctrlR_excI_olgR, "audioamplify")
		self.dctrlR_excI_olgR_wR = dctrlR_excI_olgR_wR = pipeparts.mkgeneric(self, dctrlR_excI_olgR, "audioamplify")

		dctrlI_excR = gst.element_factory_make("lal_multiplier")
		dctrlI_excR.set_property("sync", True)
		self.add(dctrlI_excR)
		pipeparts.mkqueue(self, dctrlI).link(dctrlI_excR)
		pipeparts.mkqueue(self, excR).link(dctrlI_excR)
		dctrlI_excR = pipeparts.mktee(self, dctrlI_excR)

		self.dctrlI_excR_olgI = dctrlI_excR_olgI = pipeparts.mkgeneric(self, dctrlI_excR, "audioamplify")
		dctrlI_excR_olgI = pipeparts.mktee(self, dctrlI_excR_olgI)
		self.dctrlI_excR_olgI_wR = dctrlI_excR_olgI_wR = pipeparts.mkgeneric(self, dctrlI_excR_olgI, "audioamplify")
		self.dctrlI_excR_olgI_wI = dctrlI_excR_olgI_wI = pipeparts.mkgeneric(self, dctrlI_excR_olgI, "audioamplify")

		self.dctrlI_excR_olgR = dctrlI_excR_olgR = pipeparts.mkgeneric(self, dctrlI_excR, "audioamplify")
		dctrlI_excR_olgR = pipeparts.mktee(self, dctrlI_excR_olgR)
		self.dctrlI_excR_olgR_wI = dctrlI_excR_olgR_wI = pipeparts.mkgeneric(self, dctrlI_excR_olgR, "audioamplify")
		self.dctrlI_excR_olgR_wR = dctrlI_excR_olgR_wR = pipeparts.mkgeneric(self, dctrlI_excR_olgR, "audioamplify")

		dctrlI_excI = gst.element_factory_make("lal_multiplier")
		dctrlI_excI.set_property("sync", True)
		self.add(dctrlI_excI)
		pipeparts.mkqueue(self, dctrlI).link(dctrlI_excI)
		pipeparts.mkqueue(self, excI).link(dctrlI_excI)
		dctrlI_excI = pipeparts.mktee(self, dctrlI_excI)

		self.dctrlI_excI_olgR = dctrlI_excI_olgR = pipeparts.mkgeneric(self, dctrlI_excI, "audioamplify")
		dctrlI_excI_olgR = pipeparts.mktee(self, dctrlI_excI_olgR)
		self.dctrlI_excI_olgR_wR = dctrlI_excI_olgR_wR = pipeparts.mkgeneric(self, dctrlI_excI_olgR, "audioamplify")
		self.dctrlI_excI_olgR_wI = dctrlI_excI_olgR_wI = pipeparts.mkgeneric(self, dctrlI_excI_olgR, "audioamplify")

		self.dctrlI_excI_olgI = dctrlI_excI_olgI = pipeparts.mkgeneric(self, dctrlI_excI, "audioamplify")
		dctrlI_excI_olgI = pipeparts.mktee(self, dctrlI_excI_olgI)
		self.dctrlI_excI_olgI_wI = dctrlI_excI_olgI_wI = pipeparts.mkgeneric(self, dctrlI_excI_olgI, "audioamplify")
		self.dctrlI_excI_olgI_wR = dctrlI_excI_olgI_wR = pipeparts.mkgeneric(self, dctrlI_excI_olgI, "audioamplify")

		dctrlR_excR = gst.element_factory_make("lal_multiplier")
		dctrlR_excR.set_property("sync", True)
		self.add(dctrlR_excR)
		pipeparts.mkqueue(self, dctrlR).link(dctrlR_excR)
		pipeparts.mkqueue(self, excR).link(dctrlR_excR)
		dctrlR_excR = pipeparts.mktee(self, dctrlR_excR)

		self.dctrlR_excR_olgR = dctrlR_excR_olgR = pipeparts.mkgeneric(self, dctrlR_excR, "audioamplify")
		dctrlR_excR_olgR = pipeparts.mktee(self, dctrlR_excR_olgR)
		self.dctrlR_excR_olgR_wR = dctrlR_excR_olgR_wR = pipeparts.mkgeneric(self, dctrlR_excR_olgR, "audioamplify")
		self.dctrlR_excR_olgR_wI = dctrlR_excR_olgR_wI = pipeparts.mkgeneric(self, dctrlR_excR_olgR, "audioamplify")

		self.dctrlR_excR_olgI = dctrlR_excR_olgI = pipeparts.mkgeneric(self, dctrlR_excR, "audioamplify")
		dctrlR_excR_olgI = pipeparts.mktee(self, dctrlR_excR_olgI)
		self.dctrlR_excR_olgI_wI = dctrlR_excR_olgI_wI = pipeparts.mkgeneric(self, dctrlR_excR_olgI, "audioamplify")
		self.dctrlR_excR_olgI_wR = dctrlR_excR_olgI_wR = pipeparts.mkgeneric(self, dctrlR_excR_olgI, "audioamplify")

		# Combine all of these branches into real and imaginary gamma
		denom = pipeparts.mktee(self, dctrl_mod_w_mod_olg_mod)

		gammaI_num = gst.element_factory_make("lal_adder")
		gammaI_num.set_property("sync", True)
		self.add(gammaI_num)
		pipeparts.mkqueue(self, dctrl_mod_w_mod_olgI).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlR_excI_olgI_wI, -1.0)).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlI_excR_olgI_wI).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excI_olgR_wI, -1.0)).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlR_excR_olgR_wI, -1.0)).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excI_olgI_wR, -1.0)).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlR_excR_olgI_wR, -1.0)).link(gammaI_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgR_wR).link(gammaI_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excR_olgR_wR, -1.0)).link(gammaI_num)

		gammaR_num = gst.element_factory_make("lal_adder")
		gammaR_num.set_property("sync", True)
		self.add(gammaR_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrl_mod_w_mod_olgR, -1.0)).link(gammaR_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excI_olgI_wI, -1.0)).link(gammaR_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlR_excR_olgI_wI, -1.0)).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgR_wI).link(gammaR_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excR_olgR_wI, -1.0)).link(gammaR_num)
		pipeparts.mkqueue(self, dctrlR_excI_olgI_wR).link(gammaR_num)
		pipeparts.mkqueue(self, pipeparts.mkaudioamplify(self, dctrlI_excR_olgI_wR, -1.0)).link(gammaR_num)
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
