# Copyright (C) 2017  Kipp Cannon
# Copyright (C) 2011--2014  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2013  Jacob Peoples
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


import cmath
try:
	from fpconst import NegInf
except ImportError:
	# not all machines have fpconst installed
	NegInf = float("-inf")
import math
import numpy
import os
import random
from scipy import interpolate
from scipy import stats
import sys
import warnings
import json


from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo import segments
from gstlal import chirptime
from gstlal.stats import horizonhistory
from gstlal.stats import inspiral_extrinsics
from gstlal.stats import inspiral_intrinsics
from gstlal.stats import trigger_rate
import lal
from lal import rate
from lalburst import snglcoinc
import lalsimulation


# FIXME:  caution, this information might get organized differently later.
# for now we just need to figure out where the gstlal-inspiral directory in
# share/ is.  don't write anything that assumes that this module will
# continue to define any of these symbols
from gstlal import paths as gstlal_config_paths


__all__ = [
	"LnSignalDensity",
	"LnNoiseDensity",
	"DatalessLnSignalDensity",
	"DatalessLnNoiseDensity",
	"OnlineFrankensteinLnSignalDensity",
	"OnlineFrankensteinLnNoiseDensity"
]


# The typical horizon distance used to normalize such values in the likelihood
# ratio. Units are Mpc
TYPICAL_HORIZON_DISTANCE = 150.


# utility function to make an idq interpolator from an h5 file
def idq_interp(idq_file):
	# set a default function to simply return 0
	out = {}
	for ifo in ("H1", "L1", "V1", "K1"):
		out[ifo] = lambda x: numpy.zeros(len(x))
	if idq_file is not None:
		import h5py
		from scipy.interpolate import interp1d
		f = h5py.File(idq_file, "r")
		for ifo in f:
			# Pad the boundaries - FIXME remove
			t = [0.] + list(numpy.array(f[ifo]["time"])) + [2000000000.]
			d = [0.] + list(numpy.array(f[  ifo]["data"])) + [0.]
			out[ifo] = interp1d(t, d)
	return out


#
# =============================================================================
#
#                                  Base Class
#
# =============================================================================
#


class LnLRDensity(snglcoinc.LnLRDensity):
	# range of SNRs covered by this object
	snr_min = 4.0
	chi2_over_snr2_min = 1e-5
	chi2_over_snr2_max = 1e20
	bankchi2_over_snr2_min = 1e-5
	bankchi2_over_snr2_max = 1e20

	# SNR, \chi^2 binning definition
	snr_chi_binning = rate.NDBins((rate.ATanLogarithmicBins(2.6, 26., 300), rate.ATanLogarithmicBins(.001, 0.2, 280)))
	snr_bankchi_binning = rate.NDBins((rate.ATanLogarithmicBins(2.6, 26., 300), rate.ATanLogarithmicBins(.001, 0.2, 280)))
	chi_bankchi_binning = rate.NDBins((rate.ATanLogarithmicBins(.001, 0.2, 280), rate.ATanLogarithmicBins(.001, 0.2, 280)))
	chi_ratio_binning = rate.NDBins((rate.ATanLogarithmicBins(.001, 0.2, 280), rate.ATanLogarithmicBins(0.1, 10, 280)))

	def __init__(self, template_ids, instruments, delta_t, min_instruments = 2):
		#
		# check input
		#

		if template_ids is not None and not template_ids:
			raise ValueError("template_ids cannot be empty")
		if min_instruments < 1:
			raise ValueError("min_instruments=%d must be >= 1" % min_instruments)
		if min_instruments > len(instruments):
			raise ValueError("not enough instruments (%s) to satisfy min_instruments=%d" % (", ".join(sorted(instruments)), min_instruments))
		if delta_t < 0.:
			raise ValueError("delta_t=%g must be >= 0" % delta_t)

		#
		# initialize
		#

		self.template_ids = frozenset(template_ids) if template_ids is not None else template_ids
		self.instruments = frozenset(instruments)
		self.min_instruments = min_instruments
		self.delta_t = delta_t
		# install SNR, chi^2 PDFs.  numerator will override this
		self.densities = {}
		for instrument in instruments:
			self.densities["%s_snr_chi" % instrument] = rate.BinnedLnPDF(self.snr_chi_binning)
			self.densities["%s_snr_bankchi" % instrument] = rate.BinnedLnPDF(self.snr_bankchi_binning)
			self.densities["%s_chi_bankchi" % instrument] = rate.BinnedLnPDF(self.chi_bankchi_binning)
			self.densities["%s_chi_ratio" % instrument] = rate.BinnedLnPDF(self.chi_ratio_binning)
			#FIXME finish method is broken and won't make correct kernel for chi/bankchi or
			# chi/ratio histograms. This should be fine until we incorporate in LR.

	def __iadd__(self, other):
		if type(other) != type(self):
			raise TypeError(other)
		# template_id set mismatch is allowed in the special case
		# that one or the other is None to make it possible to
		# construct generic seed objects providing initialization
		# data for the ranking statistics.
		if self.template_ids is not None and other.template_ids is not None and self.template_ids != other.template_ids:
			raise ValueError("incompatible template IDs")
		if self.instruments != other.instruments:
			raise ValueError("incompatible instrument sets")
		if self.min_instruments != other.min_instruments:
			raise ValueError("incompatible minimum number of instruments")
		if self.delta_t != other.delta_t:
			raise ValueError("incompatible delta_t coincidence thresholds")

		if self.template_ids is None and other.template_ids is not None:
			self.template_ids = frozenset(other.template_ids)

		for key, lnpdf in self.densities.items():
			lnpdf += other.densities[key]

		try:
			del self.interps
		except AttributeError:
			pass
		return self

	def increment(self, event):
		# this test is intended to fail if .template_ids is None:
		# must not collect trigger statistics unless we can verify
		# that they are for the correct templates.
		if event.template_id not in self.template_ids:
			raise ValueError("event from wrong template")
		# ignore events below threshold.  the trigger generation
		# mechanism now produces sub-threshold triggers to
		# facilitiate the construction of sky maps in the online
		# search.  it's therefore now our responsibility to exclude
		# them from the PDFs in the ranking statistic here.
		#
		# FIXME:  this creates an inconsistency.  we exclude events
		# from the PDFs, but we don't exclude them from the
		# evaluation of those PDFs in the .__call__() method,
		# instead we'll report that the candidate is impossible
		# because it contains a trigger in a "forbidden" region of
		# the parameter space (forbidden because we've ensured
		# nothing populates those bins).  the .__call__() methods
		# should probably be modified to exclude triggers with
		# sub-threshold SNRs just as we've excluded them here, but
		# I'm not yet sure.  instead, for the time being, we
		# resolve the problem by excluding sub-threshold triggers
		# from the "-from-triggers" method in far.py.  there is a
		# FIXME related to this same issue in that code.
		if event.snr < self.snr_min:
			return
		self.densities["%s_snr_chi" % event.ifo].count[event.snr, event.chisq / event.snr**2.] += 1.0
		self.densities["%s_snr_bankchi" % event.ifo].count[event.snr, event.bank_chisq / event.snr**2.] += 1.0
		self.densities["%s_chi_bankchi" % event.ifo].count[event.chisq / event.snr**2., event.bank_chisq / event.snr**2.] += 1.0
		self.densities["%s_chi_ratio" % event.ifo].count[event.chisq / event.snr**2., event.bank_chisq / event.chisq] += 1.0

	def copy(self):
		new = type(self)(self.template_ids, self.instruments, self.delta_t, self.min_instruments)
		for key, lnpdf in self.densities.items():
			new.densities[key] = lnpdf.copy()
		return new

	def mkinterps(self):
		self.interps = dict((key, lnpdf.mkinterp()) for key, lnpdf in self.densities.items())

	def finish(self):
		snr_kernel_width_at_8 = 8.,
		chisq_kernel_width = 0.08,
		sigma = 10.
		for key, lnpdf in self.densities.items():
			# construct the density estimation kernel.  be
			# extremely conservative and assume only 1 in 10
			# samples are independent, but assume there are
			# always at least 1e7 samples.
			numsamples = max(lnpdf.array.sum() / 10. + 1., 1e6)
			snr_bins, chisq_bins = lnpdf.bins
			snr_per_bin_at_8 = (snr_bins.upper() - snr_bins.lower())[snr_bins[8.]]
			chisq_per_bin_at_0_02 = (chisq_bins.upper() - chisq_bins.lower())[chisq_bins[0.02]]

			# apply Silverman's rule so that the width scales
			# with numsamples**(-1./6.) for a 2D PDF
			snr_kernel_bins = snr_kernel_width_at_8 / snr_per_bin_at_8 / numsamples**(1./6.)
			chisq_kernel_bins = chisq_kernel_width / chisq_per_bin_at_0_02 / numsamples**(1./6.)

			# check the size of the kernel. We don't ever let
			# it get smaller than the 2.5 times the bin size
			if snr_kernel_bins < 2.5:
				snr_kernel_bins = 2.5
				warnings.warn("Replacing snr kernel bins with 2.5")
			if chisq_kernel_bins < 2.5:
				chisq_kernel_bins = 2.5
				warnings.warn("Replacing chisq kernel bins with 2.5")

			# convolve bin count with density estimation kernel
			rate.filter_array(lnpdf.array, rate.gaussian_window(snr_kernel_bins, chisq_kernel_bins, sigma = sigma))

			# zero everything below the SNR cut-off.  need to
			# do the slicing ourselves to avoid zeroing the
			# at-threshold bin
			lnpdf.array[:lnpdf.bins[0][self.snr_min],:] = 0.

			# normalize what remains
			lnpdf.normalize()
		self.mkinterps()

		#
		# never allow PDFs that have had the density estimation
		# transform applied to be written to disk:  on-disk files
		# must only ever provide raw counts.  also don't allow
		# density estimation to be applied twice
		#

		def to_xml(*args, **kwargs):
			raise NotImplementedError("writing .finish()'ed LnLRDensity object to disk is forbidden")
		self.to_xml = to_xml
		def finish(*args, **kwargs):
			raise NotImplementedError(".finish()ing a .finish()ed LnLRDensity object is forbidden")
		self.finish = finish

	def to_xml(self, name):
		xml = super(LnLRDensity, self).to_xml(name)
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"template_ids", ",".join("%d" % template_id for template_id in sorted(self.template_ids)) if self.template_ids is not None else None))
		# FIXME this is not an ideal way to get only one into the file
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"instruments", lsctables.instrumentsproperty.set(self.instruments)))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"min_instruments", self.min_instruments))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"delta_t", self.delta_t))
		for key, lnpdf in self.densities.items():
			# never write PDFs to disk without ensuring the
			# normalization metadata is up to date
			lnpdf.normalize()
			xml.appendChild(lnpdf.to_xml(key))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		template_ids = ligolw_param.get_pyvalue(xml, u"template_ids")
		# FIXME find a better way to do this: Allow a file to not have
		# a mass model filename.  This is can happen when e.g., gstlal
		# inspiral produces a ranking stat file which just records but
		# doesn't evaluate LRs.  Then you marginalize it in with a
		# create prior diststats file which does have the mass model.
		# The iadd method makes sure that the mass model is unique or
		# that it doesn't exist.
		if template_ids is not None:
			template_ids = frozenset(int(i) for i in template_ids.split(","))
		self = cls(
			template_ids = template_ids,
			instruments = lsctables.instrumentsproperty.get(ligolw_param.get_pyvalue(xml, u"instruments")),
			delta_t = ligolw_param.get_pyvalue(xml, u"delta_t"),
			min_instruments = ligolw_param.get_pyvalue(xml, u"min_instruments")
		)
		for key in self.densities:
			self.densities[key] = self.densities[key].from_xml(xml, key)
		return self


#
# =============================================================================
#
#                              Numerator Density
#
# =============================================================================
#


class LnSignalDensity(LnLRDensity):
	def __init__(self, *args, **kwargs):
		self.population_model_file = kwargs.pop("population_model_file", None)
		self.dtdphi_file = kwargs.pop("dtdphi_file", None)
		self.horizon_factors = kwargs.pop("horizon_factors", None)
		self.idq_file = kwargs.pop("idq_file", None)
		super(LnSignalDensity, self).__init__(*args, **kwargs)
		# install SNR, chi^2 PDF (one for all instruments)
		self.densities = {
			"snr_chi": inspiral_extrinsics.NumeratorSNRCHIPDF(self.snr_chi_binning)
		}

		# record of horizon distances for all instruments in the
		# network
		self.horizon_history = horizonhistory.HorizonHistories((instrument, horizonhistory.NearestLeafTree()) for instrument in self.instruments)
		# source population model
		# FIXME:  introduce a mechanism for selecting the file
		self.population_model = inspiral_intrinsics.SourcePopulationModel(self.template_ids, filename = self.population_model_file)
		self.InspiralExtrinsics = inspiral_extrinsics.InspiralExtrinsics(self.min_instruments, filename = self.dtdphi_file)

		# initialize an IDQ likelihood of glitch interpolator
		self.idq_glitch_lnl = idq_interp(self.idq_file)

	def set_horizon_factors(self, horizon_factors):
		self.horizon_factors = horizon_factors

	def __call__(self, segments, snrs, chi2s_over_snr2s, phase, dt, template_id):
		assert frozenset(segments) == self.instruments
		if len(snrs) < self.min_instruments:
			return NegInf

		# use volume-weighted average horizon distance over
		# duration of event to estimate sensitivity
		assert all(segments.values()), "encountered trigger with duration = 0"
		horizons = dict((instrument, (self.horizon_history[instrument].functional_integral(map(float, seg), lambda d: d**3.) / float(abs(seg)))**(1./3.)) for instrument, seg in segments.items())

		# compute P(t).  P(t) \propto volume within which a signal will
		# produce a candidate * number of trials \propto cube of
		# distance to which the mininum required number of instruments
		# can see (I think) * number of templates.  we measure the
		# distance in multiples of TYPICAL_HORIZON_DISTANCE Mpc just to
		# get a number that will be, typically, a little closer to 1,
		# in lieu of properly normalizating this factor.  we can't
		# normalize the factor because the normalization depends on the
		# duration of the experiment, and that keeps changing when
		# running online, combining offline chunks from different
		# times, etc., and that would prevent us from being able to
		# compare a ln L ranking statistic value defined during one
		# period to ranking statistic values defined in other periods.
		# by removing the normalization, and leaving this be simply a
		# factor that is proportional to the rate of signals, we can
		# compare ranking statistics across analysis boundaries but we
		# loose meaning in the overall scale:  ln L = 0 is not a
		# special value, as it would be if the numerator and
		# denominator were both normalized properly.
		horizon = sorted(horizons.values())[-self.min_instruments] / TYPICAL_HORIZON_DISTANCE
		if not horizon:
			return NegInf
		# horizon factors adjusts the computed horizon factor according
		# to the sigma values computed at the time of the SVD. This
		# gives a good approximation to the horizon distance for each
		# template without computing them each explicitly. Only one
		# template has its horizon calculated explicitly.
		lnP = 3. * math.log(horizon * self.horizon_factors[template_id]) + math.log(len(self.template_ids))

		# Add P(instruments | horizon distances)
		try:
			lnP += math.log(self.InspiralExtrinsics.p_of_instruments_given_horizons(snrs.keys(), horizons))
		except ValueError:
			# The code raises a value error when a needed horizon distance is zero
			return NegInf

		# Evaluate dt, dphi, snr probability
		try:
			lnP += math.log(self.InspiralExtrinsics.time_phase_snr(dt, phase, snrs, horizons))
		# FIXME need to make sure this is really a math domain error
		except ValueError:
			return NegInf

		# evaluate population model
		lnP += self.population_model.lnP_template_signal(template_id, max(snrs.values()))

		# evalute the (snr, \chi^2 | snr) PDFs (same for all
		# instruments)
		interp = self.interps["snr_chi"]

		# Evaluate the IDQ glitch probability # FIXME put in denominator
		for ifo, seg in segments.items():
			# only proceed if we have a trigger from this ifo
			if ifo in snrs:
				# FIXME don't just use the last segment, somehow include whole template duration?
				t = float(seg[1])
				# NOTE choose max over +-1 seconds because the sampling is only at 1 Hz.
				lnP -= max(self.idq_glitch_lnl[ifo]([t-1., t, t+1.]))

		return lnP + sum(interp(snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def __iadd__(self, other):
		super(LnSignalDensity, self).__iadd__(other)
		self.horizon_history += other.horizon_history
		if self.population_model_file is not None and other.population_model_file is not None and other.population_model_file != self.population_model_file:
			raise ValueError("incompatible mass model file names")
		if self.population_model_file is None and other.population_model_file is not None:
			self.population_model_file = other.population_model_file
		if self.dtdphi_file is not None and other.dtdphi_file is not None and other.dtdphi_file != self.dtdphi_file:
			raise ValueError("incompatible dtdphi files")
		if self.dtdphi_file is None and other.dtdphi_file is not None:
			self.dtdphi_file = other.dtdphi_file
		if self.idq_file is not None and other.idq_file is not None and other.idq_file != self.idq_file:
			raise ValueError("incompatible idq files")
		if self.idq_file is None and other.idq_file is not None:
			self.idq_file = other.idq_file
		if self.horizon_factors is not None and other.horizon_factors is not None and other.horizon_factors != self.horizon_factors:
			# require that the horizon factors be the same within 1%
			for k in self.horizon_factors:
				if 0.99 > self.horizon_factors[k] / other.horizon_factors[k] > 1.01:
					raise ValueError("incompatible horizon_factors")
		if self.horizon_factors is None and other.horizon_factors is not None:
			self.horizon_factors = other.horizon_factors

		return self

	def increment(self, *args, **kwargs):
		raise NotImplementedError

	def copy(self):
		new = super(LnSignalDensity, self).copy()
		new.horizon_history = self.horizon_history.copy()
		new.population_model_file = self.population_model_file
		new.dtdphi_file = self.dtdphi_file
		new.idq_file = self.idq_file
		# okay to use references because read-only data
		new.population_model = self.population_model
		new.InspiralExtrinsics = self.InspiralExtrinsics
		new.horizon_factors = self.horizon_factors
		new.idq_glitch_lnl = self.idq_glitch_lnl
		return new

	def local_mean_horizon_distance(self, gps, window = segments.segment(-32., +2.)):
		# horizon distance window is in seconds.  this is sort of a
		# hack, we should really tie this to each waveform's filter
		# length somehow, but we don't have a way to do that and
		# it's not clear how much would be gained by going to the
		# trouble of implementing that.  I don't even know what to
		# set it to, so for now it's set to something like the
		# typical width of the whitening filter's impulse response.
		t = abs(window)
		vtdict = self.horizon_history.functional_integral_dict(window.shift(float(gps)), lambda D: D**3.)
		return dict((instrument, (vt / t)**(1./3.)) for instrument, vt in vtdict.items())

	def add_signal_model(self, prefactors_range = (0.001, 0.30), df = 100, inv_snr_pow = 4.):
		# normalize to 10 *mi*llion signals.  this count makes the
		# density estimation code choose a suitable kernel size
		inspiral_extrinsics.NumeratorSNRCHIPDF.add_signal_model(self.densities["snr_chi"], 1e12, prefactors_range, df, inv_snr_pow = inv_snr_pow, snr_min = self.snr_min)
		self.densities["snr_chi"].normalize()

	def candidate_count_model(self, rate = 1000.):
		"""
		Compute and return a prediction for the total number of
		above-threshold signal candidates expected.  The rate
		parameter sets the nominal signal rate in units of Gpc^-3
		a^-1.
		"""
		# FIXME:  this needs to understand a mass distribution
		# model and what part of the mass space this numerator PDF
		# is for
		seg = (self.horizon_history.minkey(), self.horizon_history.maxkey())
		V_times_t = self.horizon_history.functional_integral(seg, lambda horizons: sorted(horizons.values())[-self.min_instruments]**3.)
		# Mpc**3 --> Gpc**3, seconds --> years
		V_times_t *= 1e-9 / (86400. * 365.25)
		return V_times_t * rate * len(self.template_ids)

	def random_sim_params(self, sim, template_id = None, snr_efficiency = 1.0, f_low = 15.):
		"""
		Generator that yields an endless sequence of randomly
		generated parameter dictionaries drawn from the
		distribution of parameters expected for the given
		injection, which is an instance of a SimInspiral table row
		object (see ligo.lw.lsctables.SimInspiral for more
		information).  Each value in the sequence is a tuple, the
		first element of which is the random parameter dictionary
		and the second is 0.

		See also:

		LnNoiseDensity.random_params()

		The parameters do not necessarily represent valid
		candidates, for example the number of detectors reporting a
		trigger might be less than the .min_instruments parameter
		configured for the ranking statistic.  It is the job of the
		calling code to check the parameters for validity before
		using them to compute ranking statistic values.  This is
		done to allow the calling code to measure the frequency
		with which the injection is expected to be undetectable.
		Otherwise, the sequence is suitable for input to the
		.ln_lr_samples() log likelihood ratio generator.

		Bugs:

		The second element in each tuple in the sequence is merely
		a placeholder, not the natural logarithm of the PDF from
		which the sample has been drawn, as in the case of
		random_params().  Therefore, when used in combination with
		.ln_lr_samples(), the two probability densities computed
		and returned by that generator along with each log
		likelihood ratio value will simply be the probability
		densities of the signal and noise populations at that point
		in parameter space.  They cannot be used to form an
		importance weighted sampler of the log likelihood ratios.
		"""
		#
		# retrieve horizon distances from history.  NOTE:  some
		# might be zero.  FIXME:  apply template-specific
		# correction factor
		#

		horizon_distance = self.local_mean_horizon_distance(sim.time_geocent)
		instruments = set(horizon_distance)

		#
		# compute nominal SNRs.  NOTE:  some might be below
		# threshold, or even 0.
		#

		cosi = math.cos(sim.inclination)
		gmst = lal.GreenwichMeanSiderealTime(sim.time_geocent)
		snr_0 = {}
		phase_0 = {}
		# FIXME:  use horizon distance from above instead of this
		params = {u"H1":"alpha4", u"L1":"alpha5", u"V1":"alpha6"}
		for instrument, DH in horizon_distance.items():
			fp, fc = lal.ComputeDetAMResponse(lalsimulation.DetectorPrefixToLALDetector(str(instrument)).response, sim.longitude, sim.latitude, sim.polarization, gmst)
			snr_0[instrument] = getattr(sim, params[instrument])
			phase_0[instrument] = math.atan2(fc * cosi, fp * (1. + cosi**2.)/2.) - 2*sim.coa_phase + numpy.pi/2.
			phase_0[instrument] = phase_0[instrument] - math.floor((phase_0[instrument] + numpy.pi)/numpy.pi/2.)*numpy.pi*2

		#
		# compute template duration and retrieve valid template IDs
		#

		template_duration = chirptime.imr_time(f_low, sim.mass1*lal.MSUN_SI, sim.mass2*lal.MSUN_SI, numpy.sqrt(numpy.dot(sim.spin1, sim.spin1)), numpy.sqrt(numpy.dot(sim.spin2, sim.spin2)))

		template_ids = tuple(self.template_ids)

		#
		# construct SNR generators, and approximating the SNRs to
		# be fixed at the nominal SNRs construct \chi^2 generators
		#

		def snr_phase_gen(snr0, phase):
			if snr0 == 0.:
				# if the expected SNR is identically 0,
				# that indicates the instrument was off at
				# the time so the observed SNR must always
				# be 0
				while 1:
					yield 0., phase
			else:
				snr0 = cmath.rect(snr0, phase)
				randn = numpy.random.randn
				while 1:
					yield cmath.polar(snr0 + complex(*randn(2)))
		def chi2_over_snr2_gen(snr):
			# rates_lnx = ln(chi^2/snr^2)
			rates_lnx = numpy.log(self.densities["snr_chi"].bins[1].centres())
			# FIXME:  kinda broken for SNRs below self.snr_min
			# (right_hand) = lnP(chi^2/snr^2|snr)*const
			rates_cdf = self.densities["snr_chi"][max(snr, self.snr_min),:]
			assert not numpy.isnan(rates_cdf).any()
			# (right_hand) = P(<=chi^2/snr^2|snr)*const
			drcoss = self.densities["snr_chi"].bins[1].upper() - self.densities["snr_chi"].bins[1].lower()
			rates_cdf =(numpy.exp(rates_cdf)[:-1]*drcoss[:-1]).cumsum()
			assert not numpy.isnan(rates_cdf).any()
			# add a small tilt to break degeneracies so the
			# inverse function is defined everywhere then
			# normalize
			rates_cdf += numpy.linspace(0., 0.00001 * rates_cdf[-1], len(rates_cdf))
			rates_cdf /= rates_cdf[-1]
			assert not numpy.isnan(rates_cdf).any()
			# construct an interpolator for the inverse CDF
			interp = interpolate.interp1d(rates_cdf, rates_lnx[:-1])
			# draw endless random values of chi^2/snr^2 from
			# P(chi^2/snr^2|snr)
			math_exp = math.exp
			random_random = random.random
			while 1:
				yield math_exp(interp(random_random()))

		def time_gen(time):
			rvs = stats.norm(loc=time, scale = 0.002).rvs
			#rvs = stats.norm(loc=time-1.7e-3, scale = 4e-4).rvs #more accurate for BNS
			while 1:
				yield rvs()

		snr_phase = dict((instrument, iter(snr_phase_gen(snr, phase_0[instrument])).next) for instrument, snr in snr_0.items())
		#chi2s_over_snr2s = dict((instrument, iter(chi2_over_snr2_gen(instrument, snr)).next) for instrument, snr in snr_0.items())
		gens_t = dict((instrument, iter(time_gen(sim.time_at_instrument(instrument, {instrument: 0.0}))).next) for instrument in instruments)

		#
		# yield a sequence of randomly generated parameters for
		# this sim.
		#

		while 1:
			kwargs = {
				"segments":{},
				"snrs": {},
				"chi2s_over_snr2s": {},
				"phase": {},
				"dt": dict((instrument, time()) for instrument, time in gens_t.items()),
				"template_id": template_id if template_id else random.choice(template_ids)
			}
			ref_end = kwargs["dt"][min(kwargs["dt"])]
			for instrument, time in list(kwargs["dt"].items()):
				kwargs["segments"][instrument] = segments.segment(time - template_duration, time)
				kwargs["snrs"][instrument], kwargs["phase"][instrument] = snr_phase[instrument]()
				kwargs["chi2s_over_snr2s"][instrument] = chi2_over_snr2_gen(kwargs["snrs"][instrument]).next()
				kwargs["dt"][instrument] = time - ref_end
			yield kwargs

	def to_xml(self, name):
		xml = super(LnSignalDensity, self).to_xml(name)
		xml.appendChild(self.horizon_history.to_xml(u"horizon_history"))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"population_model_file", self.population_model_file))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"horizon_factors", json.dumps(self.horizon_factors) if self.horizon_factors is not None else None))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"dtdphi_file", self.dtdphi_file))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"idq_file", self.idq_file))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(LnSignalDensity, cls).from_xml(xml, name)
		self.horizon_history = horizonhistory.HorizonHistories.from_xml(xml, u"horizon_history")
		self.population_model_file = ligolw_param.get_pyvalue(xml, u"population_model_file")
		self.dtdphi_file = ligolw_param.get_pyvalue(xml, u"dtdphi_file")
		# FIXME this should be removed once there are not legacy files
		try:
			self.idq_file = ligolw_param.get_pyvalue(xml, u"idq_file")
		except ValueError as e:
			print >> sys.stderr, "idq file not found", e
			self.idq_file = None
		self.horizon_factors = ligolw_param.get_pyvalue(xml, u"horizon_factors")
		if self.horizon_factors is not None:
			# FIXME, how do we properly decode the json, I assume something in ligolw can do this?
			self.horizon_factors = self.horizon_factors.replace("\\","").replace('\\"','"')
			self.horizon_factors = json.loads(self.horizon_factors)
			self.horizon_factors = dict((int(k), v) for k, v in self.horizon_factors.items())
			assert set(self.template_ids) == set(self.horizon_factors)
		self.population_model = inspiral_intrinsics.SourcePopulationModel(self.template_ids, filename = self.population_model_file)
		self.InspiralExtrinsics = inspiral_extrinsics.InspiralExtrinsics(self.min_instruments, filename = self.dtdphi_file)
		self.idq_glitch_lnl = idq_interp(self.idq_file)
		return self


class DatalessLnSignalDensity(LnSignalDensity):
	"""
	Stripped-down version of LnSignalDensity for use in estimating
	ranking statistics when no data has been collected from an
	instrument with which to define the ranking statistic.

	Used, for example, to implement low-significance candidate culls,
	etc.

	Assumes all available instruments are on and have the same horizon
	distance, and assess candidates based only on SNR and \chi^2
	distributions.
	"""
	def __init__(self, *args, **kwargs):
		super(DatalessLnSignalDensity, self).__init__(*args, **kwargs)
		# so we're ready to go!
		self.add_signal_model()

	def __call__(self, segments, snrs, chi2s_over_snr2s, phase, dt, template_id):
		# evaluate P(t) \propto number of templates
		lnP = math.log(len(self.template_ids))

		# Add P(instruments | horizon distances).  assume all
		# instruments have TYPICAL_HORIZON_DISTANCE horizon
		# distance
		horizons = dict.fromkeys(segments, TYPICAL_HORIZON_DISTANCE)
		try:
			lnP += math.log(self.InspiralExtrinsics.p_of_instruments_given_horizons(snrs.keys(), horizons))
		except ValueError:
			# raises ValueError when a needed horizon distance
			# is zero
			return NegInf

		# Evaluate dt, dphi, snr probability
		try:
			lnP += math.log(self.InspiralExtrinsics.time_phase_snr(dt, phase, snrs, horizons))
		# FIXME need to make sure this is really a math domain error
		except ValueError:
			return NegInf

		# evaluate population model
		lnP += self.population_model.lnP_template_signal(template_id, max(snrs.values()))

		# evalute the (snr, \chi^2 | snr) PDFs (same for all
		# instruments)
		interp = self.interps["snr_chi"]
		return lnP + sum(interp(snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def __iadd__(self, other):
		raise NotImplementedError

	def increment(self, *args, **kwargs):
		raise NotImplementedError

	def copy(self):
		raise NotImplementedError

	def to_xml(self, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError

	@classmethod
	def from_xml(cls, xml, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError


class OnlineFrankensteinLnSignalDensity(LnSignalDensity):
	"""
	Version of LnSignalDensity with horizon distance history spliced in
	from another instance.  Used to solve a chicken-or-egg problem and
	assign ranking statistic values in an aonline anlysis.  NOTE:  the
	horizon history is not copied from the donor, instances of this
	class hold a reference to the donor's data, so as it is modified
	those modifications are immediately reflected here.

	For safety's sake, instances cannot be written to or read from
	files, cannot be marginalized together with other instances, nor
	accept updates from new data.
	"""
	@classmethod
	def splice(cls, src, Dh_donor):
		self = cls(src.template_ids, src.instruments, src.delta_t, population_model_file = src.population_model_file, dtdphi_file = src.dtdphi_file, idq_file = src.idq_file, min_instruments = src.min_instruments, horizon_factors = src.horizon_factors)
		for key, lnpdf in src.densities.items():
			self.densities[key] = lnpdf.copy()
		# NOTE:  not a copy.  we hold a reference to the donor's
		# data so that as it is updated, we get the updates.
		self.horizon_history = Dh_donor.horizon_history
		return self

	def __iadd__(self, other):
		raise NotImplementedError

	def increment(self, *args, **kwargs):
		raise NotImplementedError

	def copy(self):
		raise NotImplementedError

	def to_xml(self, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError

	@classmethod
	def from_xml(cls, xml, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError


#
# =============================================================================
#
#                             Denominator Density
#
# =============================================================================
#


class LnNoiseDensity(LnLRDensity):
	def __init__(self, *args, **kwargs):
		super(LnNoiseDensity, self).__init__(*args, **kwargs)

		# record of trigger counts vs time for all instruments in
		# the network
		self.triggerrates = trigger_rate.triggerrates((instrument, trigger_rate.ratebinlist()) for instrument in self.instruments)
		# point this to a LnLRDensity object containing the
		# zero-lag densities to mix zero-lag into the model.
		self.lnzerolagdensity = None

		# initialize a CoincRates object.  NOTE:  this is
		# potentially time-consuming.  the current implementation
		# includes hard-coded fast-paths for the standard
		# gstlal-based inspiral pipeline's coincidence and network
		# configurations, but if those change then doing this will
		# suck.  when scipy >= 0.19 becomes available on LDG
		# clusters this issue will go away (can use qhull's
		# algebraic geometry code for the probability
		# calculations).
		self.coinc_rates = snglcoinc.CoincRates(
			instruments = self.instruments,
			delta_t = self.delta_t,
			min_instruments = self.min_instruments
		)

	@property
	def segmentlists(self):
		return self.triggerrates.segmentlistdict()

	def __call__(self, segments, snrs, chi2s_over_snr2s, phase, dt, template_id):
		assert frozenset(segments) == self.instruments
		if len(snrs) < self.min_instruments:
			return NegInf

		# FIXME:  the +/-3600 s window thing is a temporary hack to
		# work around the problem of vetoes creating short segments
		# that have no triggers in them but that can have
		# injections recovered in them.  the +/- 3600 s window is
		# just a guess as to what might be sufficient to work
		# around it.  you might might to make this bigger.
		triggers_per_second_per_template = {}
		for instrument, seg in segments.items():
			triggers_per_second_per_template[instrument] = (self.triggerrates[instrument] & trigger_rate.ratebinlist([trigger_rate.ratebin(seg[1] - 3600., seg[1] + 3600., count = 0)])).density / len(self.template_ids)
		# sanity check rates
		assert all(triggers_per_second_per_template[instrument] for instrument in snrs), "impossible candidate in %s at %s when rates were %s triggers/s/template" % (", ".join(sorted(snrs)), ", ".join("%s s in %s" % (str(seg[1]), instrument) for instrument, seg in sorted(segments.items())), str(triggers_per_second_per_template))

		# P(t | noise) = (candidates per unit time @ t) / total
		# candidates.  by not normalizing by the total candidates
		# the return value can only ever be proportional to the
		# probability density, but we avoid the problem of the
		# ranking statistic definition changing on-the-fly while
		# running online, allowing candidates collected later to
		# have their ranking statistics compared meaningfully to
		# the values assigned to candidates collected earlier, when
		# the total number of candidates was smaller.
		lnP = math.log(sum(self.coinc_rates.strict_coinc_rates(**triggers_per_second_per_template).values()) * len(self.template_ids))

		# P(instruments | t, noise)
		lnP += self.coinc_rates.lnP_instruments(**triggers_per_second_per_template)[frozenset(snrs)]

		# evaluate dt and dphi parameters
		# NOTE: uniform and normalized so that the log should be zero, but there is no point in doing that
		# lnP += 0

		# evaluate the rest
		interps = self.interps
		return lnP + sum(interps["%s_snr_chi" % instrument](snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def __iadd__(self, other):
		super(LnNoiseDensity, self).__iadd__(other)
		self.triggerrates += other.triggerrates
		return self

	def copy(self):
		new = super(LnNoiseDensity, self).copy()
		new.triggerrates = self.triggerrates.copy()
		# NOTE:  lnzerolagdensity in the copy is reset to None by
		# this operation.  it is left as an exercise to the calling
		# code to re-connect it to the appropriate object if
		# desired.
		return new

	def mkinterps(self):
		#
		# override to mix zero-lag densities in if requested
		#

		if self.lnzerolagdensity is None:
			super(LnNoiseDensity, self).mkinterps()
		else:
			# same as parent class, but with .lnzerolagdensity
			# added
			self.interps = dict((key, (pdf + self.lnzerolagdensity.densities[key]).mkinterp()) for key, pdf in self.densities.items())

	def add_noise_model(self, number_of_events = 1):
		#
		# populate snr,chi2 binnings with a slope to force
		# higher-SNR events to be assesed to be more significant
		# when in the regime beyond the edge of measured or even
		# extrapolated background.
		#

		# pick a canonical PDF to definine the binning (we assume
		# they're all the same and only compute this array once to
		# save time
		lnpdf = self.densities.values()[0]
		arr = numpy.zeros_like(lnpdf.array)

		snrindices, rcossindices = lnpdf.bins[self.snr_min:1e10, 1e-6:1e2]
		snr, dsnr = lnpdf.bins[0].centres()[snrindices], lnpdf.bins[0].upper()[snrindices] - lnpdf.bins[0].lower()[snrindices]
		rcoss, drcoss = lnpdf.bins[1].centres()[rcossindices], lnpdf.bins[1].upper()[rcossindices] - lnpdf.bins[1].lower()[rcossindices]

		prcoss = numpy.ones(len(rcoss))
		# This adds a faint power law that falls off faster than GWs
		# but not exponential (to avoid numerical errors). It has
		# approximately the same value at SNR 8 as does exp(-8**2 / 2.)
		psnr = snr**-15
		psnr = numpy.outer(psnr, numpy.ones(len(rcoss)))
		vol = numpy.outer(dsnr, drcoss)
		# arr[snrindices, rcossindices] = psnr
		arr[snrindices, rcossindices] = psnr * vol

		# normalize to the requested count.  give 99% of the
		# requested events to this portion of the model
		arr *= number_of_events / arr.sum()

		# FIXME Handle this properly.
		for key, lnpdf in self.densities.items():
			if "chi_bankchi" not in key and "chi_ratio" not in key:
				# add in the noise model
				lnpdf.array += arr
				# re-normalize
				lnpdf.normalize()

	def candidate_count_model(self):
		"""
		Compute and return a prediction for the total number of
		noise candidates expected for each instrument
		combination.
		"""
		# assumes the trigger rate is uniformly distributed among
		# the templates and uniform in live time, calculates
		# coincidence rate assuming template exact-match
		# coincidence is required, then multiplies by the template
		# count to get total coincidence rate.
		return dict((instruments, count * len(self.template_ids)) for instruments, count in self.coinc_rates.marginalized_strict_coinc_counts(
			self.triggerrates.segmentlistdict(),
			**dict((instrument, rate / len(self.template_ids)) for instrument, rate in self.triggerrates.densities.items())
		).items())

	def random_params(self):
		"""
		Generator that yields an endless sequence of randomly
		generated candidate parameters.  NOTE: the parameters will
		be within the domain of the repsective binnings but are not
		drawn from the PDF stored in those binnings --- this is not
		an MCMC style sampler.  Each value in the sequence is a
		three-element tuple.  The first two elements of each tuple
		provide the *args and **kwargs values for calls to this PDF
		or the numerator PDF or the ranking statistic object.  The
		final is the natural logarithm (up to an arbitrary
		constant) of the PDF from which the parameters have been
		drawn evaluated at the point described by the *args and
		**kwargs.

		See also:

		random_sim_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.
		"""
		snr_slope = 0.8 / len(self.instruments)**3

		snrchi2gens = dict((instrument, iter(self.densities["%s_snr_chi" % instrument].bins.randcoord(ns = (snr_slope, 1.), domain = (slice(self.snr_min, None), slice(self.chi2_over_snr2_min, self.chi2_over_snr2_max)))).next) for instrument in self.instruments)
		t_and_rate_gen = iter(self.triggerrates.random_uniform()).next
		t_offsets_gen = dict((instruments, self.coinc_rates.plausible_toas(instruments).next) for instruments in self.coinc_rates.all_instrument_combos)
		random_randint = random.randint
		random_sample = random.sample
		random_uniform = random.uniform
		segment = segments.segment
		twopi = 2. * math.pi
		ln_1_2 = math.log(0.5)
		lnP_template_id = -math.log(len(self.template_ids))
		template_ids = tuple(self.template_ids)
		def nCk(n, k):
			return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
		while 1:
			t, rates, lnP_t = t_and_rate_gen()

			instruments = tuple(instrument for instrument, rate in rates.items() if rate > 0)
			if len(instruments) < self.min_instruments:
				# FIXME:  doing this invalidates lnP_t.  I
				# think the error is merely an overall
				# normalization error, though, and nothing
				# cares about the normalization
				continue
			# to pick instruments, we first pick an integer k
			# between m = min_instruments and n =
			# len(instruments) inclusively, then choose that
			# many unique names from among the available
			# instruments.  the probability of the outcome is
			#
			# = P(k) * P(selection | k)
			# = 1 / (n - m + 1) * 1 / nCk
			#
			# where nCk = number of k choices without
			# replacement from a collection of n things.
			k = random_randint(self.min_instruments, len(instruments))
			lnP_instruments = -math.log((len(instruments) - self.min_instruments + 1) * nCk(len(instruments), k))
			instruments = frozenset(random_sample(instruments, k))

			seq = sum((snrchi2gens[instrument]() for instrument in instruments), ())
			kwargs = {
				# FIXME: waveform duration hard-coded to
				# 10 s, generalize
				"segments": dict.fromkeys(self.instruments, segment(t - 10.0, t)),
				"snrs": dict((instrument, value[0]) for instrument, value in zip(instruments, seq[0::2])),
				"chi2s_over_snr2s": dict((instrument, value[1]) for instrument, value in zip(instruments, seq[0::2])),
				"phase": dict((instrument, random_uniform(0., twopi)) for instrument in instruments),
				# FIXME:  add dt to segments?  not
				# self-consistent if we don't, but doing so
				# screws up the test that was done to check
				# which instruments are on and off at "t"
				"dt": t_offsets_gen[instruments](),
				# FIXME random_params needs to be given a
				# meaningful template_id, but for now it is
				# not used in the likelihood-ratio
				# assignment so we don't care
				"template_id": random.choice(template_ids)
			}
			# NOTE:  I think the result of this sum is, in
			# fact, correctly normalized, but nothing requires
			# it to be (only that it be correct up to an
			# unknown constant) and I've not checked that it is
			# so the documentation doesn't promise that it is.
			# FIXME:  no, it's not normalized until the dt_dphi
			# bit is corrected for other than H1L1
			yield (), kwargs, sum(seq[1::2], lnP_t + lnP_instruments + lnP_template_id)

	def to_xml(self, name):
		xml = super(LnNoiseDensity, self).to_xml(name)
		xml.appendChild(self.triggerrates.to_xml(u"triggerrates"))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(LnNoiseDensity, cls).from_xml(xml, name)
		self.triggerrates = trigger_rate.triggerrates.from_xml(xml, u"triggerrates")
		self.triggerrates.coalesce()	# just in case
		return self


class DatalessLnNoiseDensity(LnNoiseDensity):
	"""
	Stripped-down version of LnNoiseDensity for use in estimating
	ranking statistics when no data has been collected from an
	instrument with which to define the ranking statistic.

	Used, for example, to implement low-significance candidate culls,
	etc.

	Assumes all available instruments are on and have the same horizon
	distance, and assess candidates based only on SNR and \chi^2
	distributions.
	"""

	DEFAULT_FILENAME = os.path.join(gstlal_config_paths["pkgdatadir"], "inspiral_datalesslndensity.xml.gz")

	@ligolw_array.use_in
	@ligolw_param.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass

	def __init__(self, *args, **kwargs):
		super(DatalessLnNoiseDensity, self).__init__(*args, **kwargs)

		# install SNR, chi^2 PDF (one for all instruments)
		# FIXME:  make mass dependent
		self.densities = {
			"snr_chi": rate.BinnedLnPDF.from_xml(ligolw_utils.load_filename(self.DEFAULT_FILENAME, contenthandler = self.LIGOLWContentHandler), u"datalesslnnoisedensity")
		}


	def __call__(self, segments, snrs, chi2s_over_snr2s, phase, dt, template_id):
		# assume all instruments are on, 1 trigger per second per
		# template
		triggers_per_second_per_template = dict.fromkeys(segments, 1.)

		# P(t | noise) = (candidates per unit time @ t) / total
		# candidates.  by not normalizing by the total candidates
		# the return value can only ever be proportional to the
		# probability density, but we avoid the problem of the
		# ranking statistic definition changing on-the-fly while
		# running online, allowing candidates collected later to
		# have their ranking statistics compared meaningfully to
		# the values assigned to candidates collected earlier, when
		# the total number of candidates was smaller.
		lnP = math.log(sum(self.coinc_rates.strict_coinc_rates(**triggers_per_second_per_template).values()) * len(self.template_ids))

		# P(instruments | t, noise)
		lnP += self.coinc_rates.lnP_instruments(**triggers_per_second_per_template)[frozenset(snrs)]

		# evalute the (snr, \chi^2 | snr) PDFs (same for all
		# instruments)
		interp = self.interps["snr_chi"]
		return lnP + sum(interp(snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def random_params(self):
		# won't work
		raise NotImplementedError

	def __iadd__(self, other):
		raise NotImplementedError

	def increment(self, *args, **kwargs):
		raise NotImplementedError

	def copy(self):
		raise NotImplementedError

	def to_xml(self, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError

	@classmethod
	def from_xml(cls, xml, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError


class OnlineFrankensteinLnNoiseDensity(LnNoiseDensity):
	"""
	Version of LnNoiseDensity with trigger rate data spliced in from
	another instance.  Used to solve a chicken-or-egg problem and
	assign ranking statistic values in an aonline anlysis.  NOTE:  the
	trigger rate data is not copied from the donor, instances of this
	class hold a reference to the donor's data, so as it is modified
	those modifications are immediately reflected here.

	For safety's sake, instances cannot be written to or read from
	files, cannot be marginalized together with other instances, nor
	accept updates from new data.
	"""
	@classmethod
	def splice(cls, src, rates_donor):
		self = cls(src.template_ids, src.instruments, src.delta_t, min_instruments = src.min_instruments)
		for key, lnpdf in src.densities.items():
			self.densities[key] = lnpdf.copy()
		# NOTE:  not a copy.  we hold a reference to the donor's
		# data so that as it is updated, we get the updates.
		self.triggerrates = rates_donor.triggerrates
		return self

	def __iadd__(self, other):
		raise NotImplementedError

	def increment(self, *args, **kwargs):
		raise NotImplementedError

	def copy(self):
		raise NotImplementedError

	def to_xml(self, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError

	@classmethod
	def from_xml(cls, xml, name):
		# I/O not permitted:  the on-disk version would be
		# indistinguishable from a real ranking statistic and could
		# lead to accidents
		raise NotImplementedError
