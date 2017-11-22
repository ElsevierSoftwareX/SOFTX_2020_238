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

## @file
# The python module to implement false alarm probability and false alarm rate
#
# ### Review Status
#
# STATUS: reviewed with actions
#
# | Names                                                                                 | Hash                                        | Date       | Diff to Head of Master      |
# | -------------------------------------------                                           | ------------------------------------------- | ---------- | --------------------------- |
# | Hanna, Cannon, Meacher, Creighton J, Robinet, Sathyaprakash, Messick, Dent, Blackburn | 7fb5f008afa337a33a72e182d455fdd74aa7aa7a | 2014-11-05 |<a href="@gstlal_inspiral_cgit_diff/python/far.py?id=HEAD&id2=7fb5f008afa337a33a72e182d455fdd74aa7aa7a">far.py</a> |
# | Hanna, Cannon, Meacher, Creighton J, Sathyaprakash,                                   | 72875f5cb241e8d297cd9b3f9fe309a6cfe3f716 | 2015-11-06 |<a href="@gstlal_inspiral_cgit_diff/python/far.py?id=HEAD&id2=72875f5cb241e8d297cd9b3f9fe309a6cfe3f716">far.py</a> |
#
# #### Action items
#

# - Address the fixed SNR PDF using median PSD which could be pre-computed and stored on disk. (Store database of SNR pdfs for a variety of horizon)
# - The binning parameters are hard-coded too; Could it be a problem?
# - Chisquare binning hasn't been tuned to be a good representation of the PDFs; could be improved in future

## @package far


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


try:
	from fpconst import NaN, NegInf, PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	NegInf = float("-inf")
	PosInf = float("+inf")
import itertools
import math
import multiprocessing
import multiprocessing.queues
import numpy
import random
import warnings
from scipy import interpolate
from scipy import optimize
from scipy import stats
import sys


from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import segments as ligolw_segments
from glue.segmentsUtils import vote
from glue.text_progress_bar import ProgressBar
import lal
from lal import rate
from lalburst import snglcoinc
import lalsimulation


from gstlal import stats as gstlalstats
from gstlal.stats import horizonhistory
from gstlal.stats import inspiral_extrinsics


#
# =============================================================================
#
#                 Parameter Distributions Book-Keeping Object
#
# =============================================================================
#


#
# Inspiral-specific CoincParamsDistributions sub-class
#


class CoincParams(dict):
	# place-holder class to allow params dictionaries to carry
	# attributes as well
	__slots__ = ("horizons","t_offset","coa_phase")


class ThincaCoincParamsDistributions(snglcoinc.LnLikelihoodRatioMixin):
	ligo_lw_name_suffix = u"gstlal_inspiral_coincparamsdistributions"

	#
	# Default content handler for loading CoincParamsDistributions
	# objects from XML documents
	#

	@ligolw_array.use_in
	@ligolw_param.use_in
	@lsctables.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass

	# range of SNRs covered by this object
	snr_min = 4.

	# load/initialize an SNRPDF instance for use by all instances of
	# this class
	SNRPDF = inspiral_extrinsics.SNRPDF.load()
	assert SNRPDF.snr_cutoff == snr_min

	# with what weight to include the denominator PDFs in the
	# numerator.  numerator will be
	#
	# numerator =
	#   (1 - accidental_weight) * (measured numerator) +
	#   accidental_weight * (measured denominator)
	numerator_accidental_weight = 0.

	# binnings (initialized in .__init__()
	binnings = {}
	pdf_from_rates_func = {}

	def __init__(self, instruments = frozenset(("H1", "L1", "V1")), min_instruments = 2, signal_rate = 1.0, delta_t = 0.005, process_id = None, **kwargs):
		#
		# check input
		#

		assert "V1" not in instruments	# disallow Virgo from initialization FIXME:  remove after O2
		if min_instruments < 1:
			raise ValueError("min_instruments=%d must be >= 1" % min_instruments)
		if min_instruments > len(instruments):
			raise ValueError("not enough instruments (%s) to satisfy min_instruments=%d" % (", ".join(sorted(instruments)), min_instruments))
		if delta_t < 0.:
			raise ValueError("delta_t=%g must be >= 0" % delta_t)

		# in the parent class this is a class attribute, but we use
		# it as an instance attribute here
		self.binnings = dict.fromkeys(("%s_snr_chi" % instrument for instrument in instruments), rate.NDBins((rate.ATanLogarithmicBins(2.6, 26., 300), rate.ATanLogarithmicBins(.001, 0.2, 280))))
		self.binnings.update({
			"instruments": rate.NDBins((snglcoinc.InstrumentBins(instruments),)),
			"singles": rate.NDBins((rate.HashableBins(instruments),))
		})

		self.pdf_from_rates_func = dict.fromkeys(("%s_snr_chi" % instrument for instrument in instruments), self.pdf_from_rates_snrchi2)
		self.pdf_from_rates_func.update({
			"instruments": self.pdf_from_rates_instruments,
			"singles": lambda *args: None
		})

		# this can't be done until the binnings attribute has been
		# populated
		self.zero_lag_rates = dict((param, rate.BinnedArray(binning)) for param, binning in self.binnings.items())
		self.background_rates = dict((param, rate.BinnedArray(binning)) for param, binning in self.binnings.items())
		self.injection_rates = dict((param, rate.BinnedArray(binning)) for param, binning in self.binnings.items())
		self.zero_lag_pdf = {}
		self.background_pdf = {}
		self.injection_pdf = {}
		self.zero_lag_lnpdf_interp = {}
		self.background_lnpdf_interp = {}
		self.injection_lnpdf_interp = {}
		self.process_id = process_id

		# record of horizon distances for all instruments in the
		# network
		self.horizon_history = horizonhistory.HorizonHistories((instrument, horizonhistory.NearestLeafTree()) for instrument in instruments)

		# the minimum number of instruments required to form a
		# candidate
		self.min_instruments = min_instruments

		# the mean instrinsic signal rate for the region of
		# parameter space tracked by this instance.  the units are
		# arbitrary, but must be consistent across regions of
		# parameter space
		self.signal_rate = signal_rate

		# the coincidence window.  needed for modelling instrument
		# combination rates
		self.delta_t = delta_t

		# set to True to include zero-lag histograms in background model
		self.zero_lag_in_background = False

	@property
	def instruments(self):
		return frozenset(self.horizon_history)

	@staticmethod
	def addbinnedarrays(rate_target_dict, rate_source_dict, pdf_target_dict, pdf_source_dict):
		"""
		For internal use.
		"""
		weight_target = {}
		weight_source = {}
		for name, binnedarray in rate_source_dict.items():
			if name in rate_target_dict:
				weight_target[name] = rate_target_dict[name].array.sum()
				weight_source[name] = rate_source_dict[name].array.sum()
				rate_target_dict[name] += binnedarray
			else:
				rate_target_dict[name] = binnedarray.copy()
		for name, binnedarray in pdf_source_dict.items():
			if name in pdf_target_dict:
				binnedarray = binnedarray.copy()
				binnedarray.array *= weight_source[name]
				pdf_target_dict[name].array *= weight_target[name]
				pdf_target_dict[name] += binnedarray
				pdf_target_dict[name].array /= weight_source[name] + weight_target[name]
			else:
				pdf_target_dict[name] = binnedarray.copy()

	def __iadd__(self, other):
		if type(other) != type(self):
			raise TypeError(other)

		# NOTE:  because we use custom PDF constructions, the stock
		# .__iadd__() method for this class will not result in
		# valid PDFs.  the rates arrays *are* handled correctly by
		# the .__iadd__() method, by fiat, so just remember to
		# invoke .finish() to get the PDFs in shape afterwards

		# FIXME: this operation is invalid for
		# ThincaCoincParamsDistributions from different parts of
		# the parameter space.  each ThincaCoincParamsDistributions
		# should carry an identifier associating it with a template
		# bank bin and this method should refuse to add instances
		# from different bins unless some sort of --force override
		# is enabled.  for now we rely on the pipeline's workflow
		# code to not do the wrong thing.
		if self.instruments != other.instruments:
			raise ValueError("incompatible instrument sets")
		if self.min_instruments != other.min_instruments:
			raise ValueError("incompatible minimum number of instruments")
		if self.delta_t != other.delta_t:
			raise ValueError("incompatible delta_t coincidence thresholds")

		self.addbinnedarrays(self.zero_lag_rates, other.zero_lag_rates, self.zero_lag_pdf, other.zero_lag_pdf)
		self.addbinnedarrays(self.background_rates, other.background_rates, self.background_pdf, other.background_pdf)
		self.addbinnedarrays(self.injection_rates, other.injection_rates, self.injection_pdf, other.injection_pdf)
		self.horizon_history += other.horizon_history

		#
		# rebuild interpolators
		#

		self._rebuild_interpolators()

		#
		# done
		#

		return self

	def copy(self):
		new = type(self)(process_id = self.process_id)
		new += self
		return new

	def coinc_params(self, events, offsetvector, mode = "ranking"):
		#
		# strip Virgo triggers.  FIXME:  remove after O2
		#

		assert len(events) != 0, "no triggers in candidate"
		events = tuple(event for event in events if event.ifo != "V1")
		assert len(events) != 0, "no triggers from allowed instruments in candidate"

		#
		# 2D (snr, \chi^2) values.
		#

		params = CoincParams(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events)

		#
		# instrument combination
		#

		if mode == "ranking":
			if len(events) < self.min_instruments:
				raise ValueError("candidates require >= %d events in ranking mode" % self.min_instruments)
			params["instruments"] = (frozenset(event.ifo for event in events),)
		elif mode == "counting":
			if len(events) != 1:
				raise ValueError("only singles are allowed in counting mode")
			# we don't require an offsetvector in counting mode
			# because that would be nonsensical, but we need
			# something for the code below to be happy
			if offsetvector is None:
				offsetvector = {events[0].ifo: 0.0}
		else:
			raise ValueError("invalid mode '%s'" % mode)

		#
		# record coa_phase and offset from epoch.  the epoch is
		# chosen to be the time-slid end-time of the 0th trigger.
		# the objective here is to allow the time-shifted end times
		# to be converted to floats without loss of precision in
		# such a way that singles always have an offset of 0.
		#

		params.coa_phase = dict((event.ifo, event.coa_phase) for event in events)
		ref, ref_offset = events[0].end, offsetvector[events[0].ifo]
		params.t_offset = dict((event.ifo, float(event.end - ref) + offsetvector[event.ifo] - ref_offset) for event in events)

		#
		# record the horizon distances.  pick one trigger at random
		# to provide a timestamp and pull the horizon distances
		# from our horizon distance history at that time.  the
		# horizon history is keyed by floating-point values (don't
		# need nanosecond precision for this).  NOTE:  this is
		# attached as a property instead of going into the
		# dictionary to not confuse denominator(), numerator(),
		# and friends methods.
		#
		# FIXME:  this should use .weighted_mean() to get an
		# average over a interval

		params.horizons = self.horizon_history.getdict(float(events[0].end))
		# for instruments that provided triggers,
		# use the trigger effective distance and
		# SNR to provide the horizon distance.
		# should be the same, but do this just in
		# case the history isn't as complete as
		# we'd like it to be
		#
		# FIXME:  for now this is disabled until
		# we figure out how to get itac's sigmasq
		# property updated from the whitener
		#params.horizons.update(dict((event.ifo, event.eff_distance * event.snr / 8.) for event in events))

		#
		# done
		#

		return params

	def add_zero_lag(self, param_dict, weight = 1.0):
		"""
		Increment a bin in one or more of the observed data (or
		"zero lag") histograms by weight (default 1).  The names of
		the histograms to increment, and the parameters identifying
		the bin in each histogram, are given by the param_dict
		dictionary.
		"""
		for param, value in param_dict.items():
			try:
				self.zero_lag_rates[param][value] += weight
			except IndexError:
				# param value out of range
				pass

	def add_background(self, param_dict, weight = 1.0):
		"""
		Increment a bin in one or more of the noise (or
		"background") histograms by weight (default 1).  The names
		of the histograms to increment, and the parameters
		identifying the bin in each histogram, are given by the
		param_dict dictionary.
		"""
		for param, value in param_dict.items():
			try:
				self.background_rates[param][value] += weight
			except IndexError:
				# param value out of range
				pass

	def add_injection(self, param_dict, weight = 1.0):
		"""
		Increment a bin in one or more of the signal (or
		"injection") histograms by weight (default 1).  The names
		of the histograms to increment, and the parameters
		identifying the bin in each histogram, are given by the
		param_dict dictionary.
		"""
		for param, value in param_dict.items():
			try:
				self.injection_rates[param][value] += weight
			except IndexError:
				# param value out of range
				pass

	def denominator(self, params):
		"""
		From a parameter value dictionary as returned by
		self.coinc_params(), compute and return the natural
		logarithm of the noise probability density at that point in
		parameter space.

		The .finish() method must have been invoked before this
		method does meaningful things.  No attempt is made to
		ensure the .finish() method has been invoked nor, if it has
		been invoked, that no manipulations have occured that might
		require it to be re-invoked (e.g., the contents of the
		parameter distributions have been modified and require
		re-normalization).

		This default implementation assumes the individual PDFs
		containined in the noise dictionary are for
		statistically-independent random variables, and computes
		and returns the logarithm of their product.  Sub-classes
		that require more sophisticated calculations can override
		this method.
		"""
		# evaluate dt and dphi parameters
		lnP_dt_dphi_noise = inspiral_extrinsics.lnP_dt_dphi(params, self.delta_t, model = "noise")

		# evaluate the rest
		__getitem__ = self.background_lnpdf_interp.__getitem__
		return lnP_dt_dphi_noise + sum(__getitem__(name)(*value) for name, value in params.items())

	def numerator(self, params):
		"""
		From a parameter value dictionary as returned by
		self.coinc_params(), compute and return the natural
		logarithm of the signal probability density at that point
		in parameter space.

		The .finish() method must have been invoked before this
		method does meaningful things.  No attempt is made to
		ensure the .finish() method has been invoked nor, if it has
		been invoked, that no manipulations have occured that might
		require it to be re-invoked (e.g., the contents of the
		parameter distributions have been modified and require
		re-normalization).

		This default implementation assumes the individual PDFs
		containined in the signal dictionary are for
		statistically-independent random variables, and computes
		and returns the logarithm of their product.  Sub-classes
		that require more sophisticated calculations can override
		this method.
		"""
		# NOTE:  numerator() and denominator() both omit the factor
		# P(horizon distance) = 1/T because it is identical in the
		# numerator and denominator and so factors out of the
		# ranking statistic, and because it is constant and so
		# equivalent to an irrelevant normalization factor in the
		# ranking statistic PDF sampler code.

		# instrument-->snr mapping
		snrs = dict((name.split("_", 1)[0], value[0]) for name, value in params.items() if name.endswith("_snr_chi"))
		# evaluate SNR PDF
		lnP_snr_signal = self.SNRPDF.lnP_snrs(snrs, params.horizons, self.min_instruments)

		# evaluate dt and dphi parameters
		lnP_dt_dphi_signal = inspiral_extrinsics.lnP_dt_dphi(params, self.delta_t, model = "signal")

		# FIXME:  P(instruments | signal) needs to depend on
		# horizon distances.  here we're assuming whatever
		# populate_prob_of_instruments_given_signal() has set the
		# probabilities to is OK.  we probably need to cache these
		# and save them in the XML file, too, like P(snrs | signal,
		# instruments)
		__getitem__ = self.injection_lnpdf_interp.__getitem__
		return lnP_snr_signal + lnP_dt_dphi_signal + sum(__getitem__(name)(*value) for name, value in params.items())

	def add_snrchi_prior(self, rates_dict, n, prefactors_range, df, inv_snr_pow = 4., verbose = False):
		if verbose:
			print >>sys.stderr, "synthesizing signal-like (SNR, \\chi^2) distributions ..."
		if df <= 0.:
			raise ValueError("require df >= 0: %s" % repr(df))
		if set(n) != self.instruments:
			raise ValueError("n must provide a count for exactly each of %s" % ", ".join(sorted(self.instruments)))
		#pfs = numpy.logspace(numpy.log10(prefactors_range[0]), numpy.log10(prefactors_range[1]), 100)
		pfs = numpy.linspace((prefactors_range[0]), (prefactors_range[1]), 100)
		for instrument, number_of_events in n.items():
			binarr = rates_dict["%s_snr_chi" % instrument]
			if verbose:
				progressbar = ProgressBar(instrument, max = len(pfs))
			else:
				progressbar = None

			# will need to normalize results so need new storage
			new_binarr = rate.BinnedArray(binarr.bins)

			# FIXME:  except for the low-SNR cut, the slicing
			# is done to work around various overflow and
			# loss-of-precision issues in the extreme parts of
			# the domain of definition.  it would be nice to
			# identify the causes of these and either fix them
			# or ignore them one-by-one with a comment
			# explaining why it's OK to ignore the ones being
			# ignored.  for example, computing snrchi2 by
			# exponentiating the sum of the logs of the terms
			# might permit its evaluation everywhere on the
			# domain.  can ncx2pdf() be made to work
			# everywhere?
			snrindices, rcossindices = new_binarr.bins[self.snr_min:1e10, 1e-10:1e10]
			snr, dsnr = new_binarr.bins[0].centres()[snrindices], new_binarr.bins[0].upper()[snrindices] - new_binarr.bins[0].lower()[snrindices]
			rcoss, drcoss = new_binarr.bins[1].centres()[rcossindices], new_binarr.bins[1].upper()[rcossindices] - new_binarr.bins[1].lower()[rcossindices]

			snrs2 = snr**2
			snrchi2 = numpy.outer(snrs2, rcoss) * df

			for pf in pfs:
				if progressbar is not None:
					progressbar.increment()
				new_binarr.array[snrindices, rcossindices] += gstlalstats.ncx2pdf(snrchi2, df, numpy.array([pf * snrs2]).T)

			# Add an SNR power law in with the differentials
			dsnrdchi2 = numpy.outer(dsnr / snr**inv_snr_pow, drcoss)
			new_binarr.array[snrindices, rcossindices] *= dsnrdchi2
			new_binarr.array[snrindices, rcossindices] *= number_of_events / new_binarr.array.sum()
			# add to raw counts
			binarr += new_binarr

	def add_background_prior(self, n = {"H1": 10000, "L1": 10000, "V1": 10000}, prefactors_range = (0.5, 20.), df = 40, inv_snr_pow = 2., ba = "background_rates", verbose = False):
		#
		# populate snr,chi2 binnings with a slope to force
		# higher-SNR events to be assesed to be more significant
		# when in the regime beyond the edge of measured or even
		# extrapolated background.
		#

		if set(n) != self.instruments:
			raise ValueError("n must provide a count for exactly each of %s" % ", ".join(sorted(self.instruments)))
		if verbose:
			print >>sys.stderr, "adding tilt to (SNR, \\chi^2) background PDFs ..."
		for instrument, number_of_events in n.items():
			binarr = getattr(self, ba)["%s_snr_chi" % instrument]

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)

			snrindices, rcossindices = new_binarr.bins[self.snr_min:1e10, 1e-6:1e2]
			snr, dsnr = new_binarr.bins[0].centres()[snrindices], new_binarr.bins[0].upper()[snrindices] - new_binarr.bins[0].lower()[snrindices]
			rcoss, drcoss = new_binarr.bins[1].centres()[rcossindices], new_binarr.bins[1].upper()[rcossindices] - new_binarr.bins[1].lower()[rcossindices]

			prcoss = numpy.ones(len(rcoss))
			psnr = 1e-8 * snr**-6 #(1. + 10**6) / (1. + snr**6)
			psnrdcoss = numpy.outer(numpy.exp(-(snr - 2**.5)**2/ 2.) * dsnr, numpy.exp(-(rcoss - .05)**2 / .00015*2) * drcoss)

			#new_binarr.array[snrindices, rcossindices] = numpy.outer(psnr * dsnr, prcoss * drcoss)
			new_binarr.array[snrindices, rcossindices] += psnrdcoss 
			# Normalize what's left to the requested count.
			# Give 99.999999% of the requested events to this portion of the model
			new_binarr.array *= 0.99 * number_of_events / new_binarr.array.sum()
			# add to raw counts
			getattr(self, ba)["singles"][instrument,] += number_of_events
			binarr += new_binarr
		# Give 0.00000001% of the requested events to the "glitch model"
		self.add_snrchi_prior(getattr(self, ba), dict((ifo, x * 0.01) for ifo, x in n.items()), prefactors_range = prefactors_range, df = df, inv_snr_pow = inv_snr_pow, verbose = verbose)

	def add_foreground_snrchi_prior(self, n, prefactors_range = (0.01, 0.25), df = 40, inv_snr_pow = 4., verbose = False):
		self.add_snrchi_prior(self.injection_rates, n, prefactors_range, df, inv_snr_pow = inv_snr_pow, verbose = verbose)

	def _rebuild_interpolators(self):
		"""
		Initialize the interp dictionaries from the discretely
		sampled PDF data.  For internal use only.
		"""
		self.zero_lag_lnpdf_interp.clear()
		self.background_lnpdf_interp.clear()
		self.injection_lnpdf_interp.clear()
		# build interpolators
		keys = set(self.zero_lag_rates)
		keys.remove("instruments")
		keys.remove("singles")
		def mkinterp(binnedarray):
			with numpy.errstate(invalid = "ignore"):
				assert not (binnedarray.array < 0.).any()
			binnedarray = binnedarray.copy()
			with numpy.errstate(divide = "ignore"):
				binnedarray.array = numpy.log(binnedarray.array)
			return rate.InterpBinnedArray(binnedarray, fill_value = NegInf)
		for key, binnedarray in self.zero_lag_pdf.items():
			if key in keys:
				self.zero_lag_lnpdf_interp[key] = mkinterp(binnedarray)
		for key, binnedarray in self.background_pdf.items():
			if key in keys:
				self.background_lnpdf_interp[key] = mkinterp(binnedarray)
		for key, binnedarray in self.injection_pdf.items():
			if key in keys:
				self.injection_lnpdf_interp[key] = mkinterp(binnedarray)

		#
		# the instrument combination "interpolators" are
		# pass-throughs.  we pre-evaluate a bunch of attribute,
		# dictionary, and method look-ups for speed
		#

		def mkinterp(binnedarray):
			binnedarray = binnedarray.copy()
			with numpy.errstate(divide = "ignore"):
				binnedarray.array = numpy.log(binnedarray.array)
			return lambda *coords: binnedarray[coords]
		if "instruments" in self.background_pdf:
			self.background_lnpdf_interp["instruments"] = mkinterp(self.background_pdf["instruments"])
		if "instruments" in self.injection_pdf:
			self.injection_lnpdf_interp["instruments"] = mkinterp(self.injection_pdf["instruments"])
		if "instruments" in self.zero_lag_pdf:
			self.zero_lag_lnpdf_interp["instruments"] = mkinterp(self.zero_lag_pdf["instruments"])

	def pdf_from_rates_instruments(self, key, pdf_dict):
		# get the binned array we're going to process
		binnedarray = pdf_dict[key]

		# optionally include zero-lag instrument combo counts in
		# the background counts
		if self.zero_lag_in_background and pdf_dict is self.background_pdf:
			binnedarray.array += self.zero_lag_rates[key].array

		# optionally mix denominator into numerator
		if pdf_dict is self.injection_pdf and self.numerator_accidental_weight:
			denom = self.background_rates[key].copy()
			if self.zero_lag_in_background:
				denom.array += self.zero_lag_rates[key].array
			for instrument in reduce(lambda a, b: a | b, denom.bins[0].containers):
				denom[frozenset([instrument]),] = 0
			binnedarray.array += denom.array * (binnedarray.array.sum() / denom.array.sum() * self.numerator_accidental_weight)

		# instrument combos are probabilities, not densities.
		with numpy.errstate(invalid = "ignore"):
			binnedarray.array /= binnedarray.array.sum()

	def pdf_from_rates_snrchi2(self, key, pdf_dict, snr_kernel_width_at_8 = 8., chisq_kernel_width = 0.08,  sigma = 10.):
		# get the binned array we're going to process
		binnedarray = pdf_dict[key]

		# optionally include zero-lag (SNR,\chi^2) counts in the
		# background counts
		if self.zero_lag_in_background and pdf_dict is self.background_pdf:
			binnedarray.array += self.zero_lag_rates[key].array

		# optionally mix denominator into numerator
		if pdf_dict is self.injection_pdf and self.numerator_accidental_weight:
			denom = self.background_rates[key].array.copy()
			if self.zero_lag_in_background:
				denom += self.zero_lag_rates[key].array
			binnedarray.array += denom * (binnedarray.array.sum() / denom.sum() * self.numerator_accidental_weight)

		numsamples = max(binnedarray.array.sum() / 10. + 1., 1e6) # Be extremely conservative and assume only 1 in 10 samples are independent, but assume there are always at least 1e7 samples.
		# construct the density estimation kernel
		snr_bins = binnedarray.bins[0]
		chisq_bins = binnedarray.bins[1]
		snr_per_bin_at_8 = (snr_bins.upper() - snr_bins.lower())[snr_bins[8.]]
		chisq_per_bin_at_0_02 = (chisq_bins.upper() - chisq_bins.lower())[chisq_bins[0.02]]

		# Apply Silverman's rule so that the width scales with numsamples**(-1./6.) for a 2D PDF
		snr_kernel_bins = snr_kernel_width_at_8 / snr_per_bin_at_8 / numsamples**(1./6.)
		chisq_kernel_bins = chisq_kernel_width / chisq_per_bin_at_0_02 / numsamples**(1./6.)

		# check the size of the kernel. We don't ever let it get
		# smaller than the 2.5 times the bin size
		if  snr_kernel_bins < 2.5:
			snr_kernel_bins = 2.5
			warnings.warn("Replacing snr kernel bins with 2.5")
		if  chisq_kernel_bins < 2.5:
			chisq_kernel_bins = 2.5
			warnings.warn("Replacing chisq kernel bins with 2.5")

		# convolve bin count with density estimation kernel
		rate.filter_array(binnedarray.array, rate.gaussian_window(snr_kernel_bins, chisq_kernel_bins, sigma = sigma))

		# zero everything below the SNR cut-off.  need to do the
		# slicing ourselves to avoid zeroing the at-threshold bin
		binnedarray.array[:binnedarray.bins[0][self.snr_min],:] = 0.

		# normalize what remains to be a valid PDF
		with numpy.errstate(invalid = "ignore"):
			binnedarray.to_pdf()

		# if this is the numerator, convert (rho, chi^2/rho^2) PDFs
		# into P(chi^2/rho^2 | rho).  don't bother unless some
		# events of this type were recorded
		if pdf_dict is self.injection_pdf and not numpy.isnan(binnedarray.array).all():
			bin_sizes = binnedarray.bins[1].upper() - binnedarray.bins[1].lower()
			for i in xrange(binnedarray.array.shape[0]):
				nonzero = binnedarray.array[i] != 0
				if not nonzero.any():
					# PDF is 0 in this column.  leave
					# that way.  result is not a valid
					# PDF, but we'll have to live with
					# it.
					continue
				norm = numpy.dot(numpy.compress(nonzero, binnedarray.array[i]), numpy.compress(nonzero, bin_sizes))
				assert not math.isnan(norm), "encountered impossible PDF:  %s is non-zero in a bin with infinite volume" % name
				binnedarray.array[i] /= norm

	def finish(self, segs, verbose = False):
		"""
		Populate the discrete PDF dictionaries from the contents of
		the rates dictionaries, and then the PDF interpolator
		dictionaries from the discrete PDFs.  The raw bin counts
		from the rates dictionaries are copied verbatim, smoothed,
		and converted to normalized PDFs using the bin volumes.
		Finally the dictionary of PDF interpolators is populated
		from the discretely sampled PDF data.
		"""
		#
		# populate background instrument combination rates
		#
		# NOTE:  we need to know the number of templates the
		# singles we have collected have come from, and we get this
		# by assuming a saturated triggering rate of 1/s, and
		# averaging the number of templates that implies across the
		# available instruments.  FIXME:  in the future, obtain the
		# number of templates from the bank
		#

		if verbose:
			print >>sys.stderr, "synthesizing background-like instrument combination probabilities ..."
		assert "V1" not in segs	# disallow Virgo.  FIXME:  remove after O2
		self.background_rates["instruments"].array[:] = 0.
		singles_counts = dict(zip(self.background_rates["singles"].bins[0].centres(), self.background_rates["singles"].array))
		num_templates = int(round(sum(singles_counts[instrument] / (float(livetime) / 1.0) for instrument, livetime in abs(segs).items()) / len(segs)))
		for instruments, count in inspiral_extrinsics.instruments_rate_given_noise(
			singles_counts = singles_counts,
			num_templates = num_templates,
			segs = segs,
			delta_t = self.delta_t,
			min_instruments = self.min_instruments,
		).items():
			self.background_rates["instruments"][instruments,] = count

		#
		# populate signal instrument combination rates  =
		# self.signal_rate * probability that a signal is
		# detectable by each of the instrument combinations.
		# because the horizon distance is 0'ed when an instrument
		# is off, this marginalization over horizon distance
		# histories also accounts for duty cycles
		#

		self.injection_rates["instruments"].array[:] = 0.
		for instruments, p in inspiral_extrinsics.P_instruments_given_signal(
			self.horizon_history,
			min_instruments = self.min_instruments
		).items():
			self.injection_rates["instruments"][instruments,] = self.signal_rate * p

		#
		# convert raw bin counts into normalized PDFs
		#

		self.zero_lag_pdf.clear()
		self.background_pdf.clear()
		self.injection_pdf.clear()
		progressbar = ProgressBar(text = "Computing Parameter PDFs", max = len(self.zero_lag_rates) + len(self.background_rates) + len(self.injection_rates)) if verbose else None
		for key, (msg, rates_dict, pdf_dict) in itertools.chain(
				zip(self.zero_lag_rates, itertools.repeat(("zero lag", self.zero_lag_rates, self.zero_lag_pdf))),
				zip(self.background_rates, itertools.repeat(("background", self.background_rates, self.background_pdf))),
				zip(self.injection_rates, itertools.repeat(("injections", self.injection_rates, self.injection_pdf)))
		):
			assert numpy.isfinite(rates_dict[key].array).all() and (rates_dict[key].array >= 0).all(), "%s %s counts are not valid" % (key, msg)
			pdf_dict[key] = rates_dict[key].copy()
			pdf_from_rates_func = self.pdf_from_rates_func[key]
			if pdf_from_rates_func is not None:
				pdf_from_rates_func(key, pdf_dict)
			if progressbar is not None:
				progressbar.increment()

		#
		# rebuild interpolators
		#

		self._rebuild_interpolators()

	@classmethod
	def get_xml_root(cls, xml, name):
		"""
		Sub-classes can use this in their overrides of the
		.from_xml() method to find the root element of the XML
		serialization.
		"""
		name = u"%s:%s" % (name, cls.ligo_lw_name_suffix)
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		return xml[0]

	@classmethod
	def from_xml(cls, xml, name):
		"""
		In the XML document tree rooted at xml, search for the
		serialized CoincParamsDistributions object named name, and
		deserialize it.  The return value is a two-element tuple.
		The first element is the deserialized
		CoincParamsDistributions object, the second is the process
		ID recorded when it was written to XML.
		"""
		# find the root element of the XML serialization
		xml = cls.get_xml_root(xml, name)

		# retrieve the process ID
		process_id = ligolw_param.get_pyvalue(xml, u"process_id")

		# create an instance
		self = cls(
			process_id = process_id,
			instruments = lsctables.instrumentsproperty.get(ligolw_param.get_pyvalue(xml, u"instruments")),
			min_instruments = ligolw_param.get_pyvalue(xml, u"min_instruments"),
			signal_rate = ligolw_param.get_pyvalue(xml, u"signal_rate"),
			delta_t = ligolw_param.get_pyvalue(xml, u"delta_t")
		)

		# reconstruct the BinnedArray objects
		def reconstruct(xml, prefix, target_dict):
			for name in [elem.Name.split(u":")[1] for elem in xml.childNodes if elem.Name.startswith(u"%s:" % prefix)]:
				target_dict[str(name)] = rate.BinnedArray.from_xml(xml, u"%s:%s" % (prefix, name))
		reconstruct(xml, u"zero_lag", self.zero_lag_rates)
		reconstruct(xml, u"zero_lag_pdf", self.zero_lag_pdf)
		reconstruct(xml, u"background", self.background_rates)
		reconstruct(xml, u"background_pdf", self.background_pdf)
		reconstruct(xml, u"injection", self.injection_rates)
		reconstruct(xml, u"injection_pdf", self.injection_pdf)

		# reconstruct horizon history
		self.horizon_history = horizonhistory.HorizonHistories.from_xml(xml, name)

		#
		# rebuild interpolators
		#

		self._rebuild_interpolators()

		#
		# done
		#

		return self

	def to_xml(self, name):
		"""
		Serialize this CoincParamsDistributions object to an XML
		fragment and return the root element of the resulting XML
		tree.  The .process_id attribute of process will be
		recorded in the serialized XML, and the object will be
		given the name name.
		"""
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"process_id", self.process_id))
		def store(xml, prefix, source_dict):
			for name, binnedarray in sorted(source_dict.items()):
				xml.appendChild(binnedarray.to_xml(u"%s:%s" % (prefix, name)))
		store(xml, u"zero_lag", self.zero_lag_rates)
		store(xml, u"zero_lag_pdf", self.zero_lag_pdf)
		store(xml, u"background", self.background_rates)
		store(xml, u"background_pdf", self.background_pdf)
		store(xml, u"injection", self.injection_rates)
		store(xml, u"injection_pdf", self.injection_pdf)

		xml.appendChild(ligolw_param.Param.from_pyvalue(u"instruments", lsctables.instrumentsproperty.set(self.instruments)))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"min_instruments", self.min_instruments))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"signal_rate", self.signal_rate))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"delta_t", self.delta_t))
		xml.appendChild(self.horizon_history.to_xml(name))
		return xml

	def random_params(self, instruments):
		"""
		Generator that yields an endless sequence of randomly
		generated parameter dictionaries for the given instruments.
		NOTE: the parameters will be within the domain of the
		repsective binnings but are not drawn from the PDF stored
		in those binnings --- this is not an MCMC style sampler.
		The return value is a tuple, the first element of which is
		the random parameter dictionary and the second is the
		natural logarithm (up to an arbitrary constant) of the PDF
		from which the parameters have been drawn evaluated at the
		co-ordinates in the parameter dictionary.

		Example:

		>>> x = iter(ThincaCoincParamsDistributions().random_params(("H1", "L1", "V1")))
		>>> x.next()

		See also:

		random_sim_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.
		"""
		if len(instruments) < self.min_instruments:
			raise ValueError("cannot simulate candidates for < %d instruments" % self.min_instruments)
		assert "V1" not in instruments	# disallow Virgo.  FIXME:  remove after O2

		snr_slope = 0.8 / len(instruments)**3

		keys = tuple("%s_snr_chi" % instrument for instrument in instruments)
		base_params = {"instruments": (frozenset(instruments),)}
		horizongen = iter(self.horizon_history.randhorizons()).next
		# P(horizons) = 1/livetime
		# FIXME:  the dt portion is only correct for the H1L1 pair
		log_P_horizons_dt_dphi = -math.log(self.horizon_history.maxkey() - self.horizon_history.minkey()) + inspiral_extrinsics.lnP_dt_dphi_uniform_H1L1(self.delta_t)
		coordgens = tuple(iter(self.binnings[key].randcoord(ns = (snr_slope, 1.), domain = (slice(self.snr_min, None), slice(None, None)))).next for key in keys)
		t_offsets_gen = snglcoinc.CoincSynthesizer(segmentlists = dict.fromkeys(self.instruments, None), delta_t = self.delta_t).plausible_toas(instruments)
		random_uniform = random.uniform
		twopi = 2. * math.pi
		while 1:
			seq = sum((coordgen() for coordgen in coordgens), ())
			params = CoincParams(zip(keys, seq[0::2]))
			params.update(base_params)
			params.t_offset = t_offsets_gen.next()
			params.coa_phase = dict((instrument, random_uniform(0., twopi)) for instrument in instruments)
			params.horizons = horizongen()
			# NOTE:  I think the result of this sum is, in
			# fact, correctly normalized, but nothing requires
			# it to be (only that it be correct up to an
			# unknown constant) and I've not checked that it is
			# so the documentation doesn't promise that it is.
			# FIXME:  no, it's not normalized until the dt_dphi
			# bit is corrected for other than H1L1
			yield params, sum(seq[1::2], log_P_horizons_dt_dphi)

	def random_sim_params(self, sim, horizon_distance = None, snr_min = None, snr_efficiency = 1.0, coinc_only = True):
		"""
		Generator that yields an endless sequence of randomly
		generated parameter dictionaries drawn from the
		distribution of parameters expected for the given
		injection, which is an instance of a SimInspiral table row
		object (see glue.ligolw.lsctables.SimInspiral for more
		information).  The return value is a tuple, the first
		element of which is the random parameter dictionary and the
		second is 0.

		See also:

		random_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.

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
		# FIXME need to add dt and dphi 
		#
		# retrieve horizon distance from history if not given
		# explicitly.  retrieve SNR threshold from class attribute
		# if not given explicitly
		#

		if horizon_distance is None:
			# FIXME:  use .weighted_mean() to get the average
			# distance over a period of time in case the
			# injection's "time" falls in an off interval
			horizon_distance = self.horizon_history[float(sim.get_time_geocent())]
		if snr_min is None:
			snr_min = self.snr_min

		#
		# compute nominal SNRs
		#

		cosi2 = math.cos(sim.inclination)**2.
		gmst = lal.GreenwichMeanSiderealTime(sim.get_time_geocent())
		snr_0 = {}
		for instrument, DH in horizon_distance.items():
			fp, fc = lal.ComputeDetAMResponse(lalsimulation.DetectorPrefixToLALDetector(str(instrument)).response, sim.longitude, sim.latitude, sim.polarization, gmst)
			snr_0[instrument] = snr_efficiency * 8. * DH * math.sqrt(fp**2. * (1. + cosi2)**2. / 4. + fc**2. * cosi2) / sim.distance

		#
		# construct SNR generators, and approximating the SNRs to
		# be fixed at the nominal SNRs construct \chi^2 generators
		#

		def snr_gen(snr):
			rvs = stats.ncx2(2., snr**2.).rvs
			math_sqrt = math.sqrt
			while 1:
				yield math_sqrt(rvs())

		def chi2_over_snr2_gen(instrument, snr):
			rates_lnx = numpy.log(self.injection_rates["%s_snr_chi" % instrument].bins[1].centres())
			# FIXME:  kinda broken for SNRs below self.snr_min
			rates_cdf = self.injection_rates["%s_snr_chi" % instrument][max(snr, self.snr_min),:].cumsum()
			# add a small tilt to break degeneracies then
			# normalize
			rates_cdf += numpy.linspace(0., 0.001 * rates_cdf[-1], len(rates_cdf))
			rates_cdf /= rates_cdf[-1]
			assert not numpy.isnan(rates_cdf).any()

			interp = interpolate.interp1d(rates_cdf, rates_lnx)
			math_exp = math.exp
			random_uniform = random.uniform
			while 1:
				yield math_exp(float(interp(random_uniform(0., 1.))))

		gens = dict(((instrument, "%s_snr_chi" % instrument), (iter(snr_gen(snr)).next, iter(chi2_over_snr2_gen(instrument, snr)).next)) for instrument, snr in snr_0.items())

		#
		# yield a sequence of randomly generated parameters for
		# this sim.
		#

		while 1:
			params = CoincParams()
			instruments = []
			for (instrument, key), (snr, chi2_over_snr2) in gens.items():
				snr = snr()
				if snr < snr_min:
					continue
				params[key] = snr, chi2_over_snr2()
				instruments.append(instrument)
			if coinc_only and len(instruments) < self.min_instruments:
				continue
			params["instruments"] = (frozenset(instruments),)
			params.horizons = horizon_distance
			yield params, 0.


#
# =============================================================================
#
#                       False Alarm Book-Keeping Object
#
# =============================================================================
#


def binned_log_likelihood_ratio_rates_from_samples(signal_rates, noise_rates, samples, nsamples):
	"""
	Populate signal and noise BinnedArray histograms from a sequence of
	samples (which can be a generator).  The first nsamples elements
	from the sequence are used.  The samples must be a sequence of
	three-element tuples (or sequences) in which the first element is a
	value of the ranking statistic (likelihood ratio) and the second
	and third elements the logs of the probabilities of obtaining that
	value of the ranking statistic in the signal and noise populations
	respectively.
	"""
	exp = math.exp
	isnan = math.isnan
	for ln_lamb, lnP_signal, lnP_noise in itertools.islice(samples, nsamples):
		if isnan(ln_lamb):
			raise ValueError("encountered NaN likelihood ratio")
		if isnan(lnP_signal) or isnan(lnP_noise):
			raise ValueError("encountered NaN signal or noise model probability densities")
		signal_rates[ln_lamb,] += exp(lnP_signal)
		noise_rates[ln_lamb,] += exp(lnP_noise)


def binned_log_likelihood_ratio_rates_from_samples_wrapper(queue, signal_rates, noise_rates, samples, nsamples):
	try:
		binned_log_likelihood_ratio_rates_from_samples(signal_rates, noise_rates, samples, nsamples)
		queue.put((signal_rates.array, noise_rates.array))
	except:
		queue.put((None, None))
		raise


#
# Class to compute ranking statistic PDFs for background-like and
# signal-like populations
#


class RankingData(object):
	ligo_lw_name_suffix = u"gstlal_inspiral_rankingdata"

	#
	# likelihood ratio binning
	#

	binnings = {
		"ln_likelihood_ratio": rate.NDBins((rate.ATanBins(0., 110., 3000),))
	}

	filters = {
		"ln_likelihood_ratio": rate.gaussian_window(4.)
	}

	def __init__(self, coinc_params_distributions, instruments = None, sampler_coinc_params_distributions = None, process_id = None, nsamples = 1000000, verbose = False):
		self.background_likelihood_rates = {}
		self.background_likelihood_pdfs = {}
		self.signal_likelihood_rates = {}
		self.signal_likelihood_pdfs = {}
		self.zero_lag_likelihood_rates = {}
		self.zero_lag_likelihood_pdfs = {}
		self.process_id = process_id

		#
		# bailout out used by .from_xml() class method to get an
		# uninitialized instance
		#

		if coinc_params_distributions is None:
			return

		#
		# initialize binnings
		#

		if coinc_params_distributions is not None:
			if instruments is not None and set(instruments) != coinc_params_distributions.instruments:
				raise ValueError("instrument set does not match coinc_params_distributions (hint:  don't supply instruments when initializing from ThincaCoincParamsDistributions)")
			instruments = tuple(coinc_params_distributions.instruments)
		elif instruments is None:
			raise ValueError("must supply an instrument set when not initializing from a ThincaCoincParamsDistributions object")
		else:
			instruments = tuple(instruments)

		for key in [frozenset(ifos) for n in range(coinc_params_distributions.min_instruments, len(instruments) + 1) for ifos in itertools.combinations(instruments, n)]:
			self.background_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])
			self.signal_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])
			self.zero_lag_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])

		#
		# run importance-weighted random sampling to populate
		# binnings.  one thread per instrument combination
		#

		if sampler_coinc_params_distributions is None:
			sampler_coinc_params_distributions = coinc_params_distributions

		threads = []
		for key in self.background_likelihood_rates:
			if verbose:
				print >>sys.stderr, "computing ranking statistic PDFs for %s" % ", ".join(sorted(key))
			q = multiprocessing.queues.SimpleQueue()
			p = multiprocessing.Process(target = lambda: binned_log_likelihood_ratio_rates_from_samples_wrapper(
				q,
				self.signal_likelihood_rates[key],
				self.background_likelihood_rates[key],
				coinc_params_distributions.ln_lr_samples(sampler_coinc_params_distributions.random_params(key), sampler_coinc_params_distributions),
				nsamples = nsamples
			))
			p.start()
			threads.append((p, q, key))
		while threads:
			p, q, key = threads.pop(0)
			self.signal_likelihood_rates[key].array, self.background_likelihood_rates[key].array = q.get()
			p.join()
			if p.exitcode:
				raise Exception("sampling thread failed")
		if verbose:
			print >>sys.stderr, "done computing ranking statistic PDFs"

		#
		# propagate knowledge of the background event rates through
		# to the ranking statistic distributions.  this information
		# is required so that when adding ranking statistic PDFs in
		# ._compute_combined_rates() or our .__iadd__() method
		# they are combined with the correct relative weights.
		# what we're doing here is making the total event count in
		# each background ranking statistic array equal to the
		# expected background coincidence event count for the
		# corresponding instrument combination.
		#

		for instruments, binnedarray in self.background_likelihood_rates.items():
			if binnedarray.array.any():
				binnedarray.array *= coinc_params_distributions.background_rates["instruments"][instruments,] / binnedarray.array.sum()

		#
		# propagate instrument combination priors through to
		# ranking statistic histograms so that
		# ._compute_combined_rates() and .__iadd__() combine the
		# histograms with the correct weights.
		#
		# FIXME:  need to also apply a weight that reflects the
		# probability of recovering a signal in the interval
		# spanned by the data these histograms reflect so that when
		# combining statistics from different intervals they are
		# summed with the correct weights.
		#

		for instruments, binnedarray in self.signal_likelihood_rates.items():
			if binnedarray.array.any():
				binnedarray.array *= coinc_params_distributions.injection_rates["instruments"][instruments,] / binnedarray.array.sum()

		#
		# compute combined rates
		#

		self._compute_combined_rates()

		#
		# populate the ranking statistic PDF arrays from the counts
		#

		self.finish()


	def collect_zero_lag_rates(self, connection, coinc_def_id):
		for instruments, ln_likelihood_ratio in connection.cursor().execute("""
SELECT
	coinc_inspiral.ifos,
	coinc_event.likelihood
FROM
	coinc_inspiral
	JOIN coinc_event ON (
		coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id
	)
WHERE
	coinc_event.coinc_def_id == ?
	AND NOT EXISTS (
		SELECT
			*
		FROM
			time_slide
		WHERE
			time_slide.time_slide_id == coinc_event.time_slide_id
			AND time_slide.offset != 0
	)
""", (coinc_def_id,)):
			assert ln_likelihood_ratio is not None, "null likelihood ratio encountered.  probably coincs have not been ranked"
			self.zero_lag_likelihood_rates[frozenset(lsctables.instrumentsproperty.get(instruments))][ln_likelihood_ratio,] += 1.

		#
		# update combined rates.  NOTE:  this recomputes all the
		# combined rates, not just the zero-lag combined rates.
		# it's safe to do this, but it might be found to be too
		# much of a performance hit every time one wants to update
		# the zero-lag rates.  if it becomes a problem, this call
		# might need to be removed from this method so that it is
		# invoked explicitly on an as-needed basis
		#

		self._compute_combined_rates()

	def _compute_combined_rates(self):
		#
		# (re-)compute combined noise and signal rates
		#

		def compute_combined_rates(rates_dict):
			try:
				del rates_dict[None]
			except KeyError:
				pass
			total_rate = rates_dict.itervalues().next().copy()
			# FIXME:  we don't bother checking that the
			# binnings are all compatible, we assume they were
			# all generated in our __init__() method and *are*
			# the same
			total_rate.array[:] = sum(binnedarray.array for binnedarray in rates_dict.values())
			rates_dict[None] = total_rate

		compute_combined_rates(self.background_likelihood_rates)
		compute_combined_rates(self.signal_likelihood_rates)
		compute_combined_rates(self.zero_lag_likelihood_rates)

	def finish(self, verbose = False):
		self.background_likelihood_pdfs.clear()
		self.signal_likelihood_pdfs.clear()
		self.zero_lag_likelihood_pdfs.clear()
		def build_pdf(binnedarray, filt):
			# copy counts into pdf array and smooth
			pdf = binnedarray.copy()
			rate.filter_array(pdf.array, filt)
			# zero the counts in the infinite-sized high bin so
			# the final PDF normalization ends up OK
			pdf.array[-1] = 0.
			# convert to normalized PDF
			pdf.to_pdf()
			return pdf
		if verbose:
			progressbar = ProgressBar(text = "Computing Log Lambda PDFs", max = len(self.background_likelihood_rates) + len(self.signal_likelihood_rates) + len(self.zero_lag_likelihood_rates))
			progressbar.show()
		else:
			progressbar = None
		for key, binnedarray in self.background_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s noise model log likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.background_likelihood_pdfs[key] = build_pdf(binnedarray, self.filters["ln_likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()
		for key, binnedarray in self.signal_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s signal model log likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.signal_likelihood_pdfs[key] = build_pdf(binnedarray, self.filters["ln_likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()
		for key, binnedarray in self.zero_lag_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s zero lag log likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.zero_lag_likelihood_pdfs[key] = build_pdf(binnedarray, self.filters["ln_likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()

	def __iadd__(self, other):
		ThincaCoincParamsDistributions.addbinnedarrays(self.background_likelihood_rates, other.background_likelihood_rates, self.background_likelihood_pdfs, other.background_likelihood_pdfs)
		ThincaCoincParamsDistributions.addbinnedarrays(self.signal_likelihood_rates, other.signal_likelihood_rates, self.signal_likelihood_pdfs, other.signal_likelihood_pdfs)
		ThincaCoincParamsDistributions.addbinnedarrays(self.zero_lag_likelihood_rates, other.zero_lag_likelihood_rates, self.zero_lag_likelihood_pdfs, other.zero_lag_likelihood_pdfs)
		return self

	@classmethod
	def new_with_extinction(cls, src):
		# create a new instance with copies of the rates arrays
		# from src
		self = cls(None, ())
		for key, binnedarray in src.background_likelihood_rates.items():
			self.background_likelihood_rates[key] = binnedarray.copy()
		for key, binnedarray in src.signal_likelihood_rates.items():
			self.signal_likelihood_rates[key] = binnedarray.copy()
		for key, binnedarray in src.zero_lag_likelihood_rates.items():
			self.zero_lag_likelihood_rates[key] = binnedarray.copy()

		# populate the combined rates, we need them for the
		# extinction model (they might have already been populated
		# in src, but this way we're sure)
		self._compute_combined_rates()

		# populate ranking statistic PDFs
		self.finish()

		# FIXME We extinct each instrument combination separately,
		# HOWEVER, only the 'None' key is used, which is all
		# instruments togther. It is an open question of whether or not
		# looking at the extinction model separately is a useful
		# diagnostic tool.  This could change in the future once we
		# have 3 instruments again to correct each by-instrument PDF
		# with the survival function
		#
		# FIXME, leaving the comment in place above, but with single
		# ifo searches it is possible (And easy) to not have enough
		# data in one. For now we just extinct None
		for key in self.background_likelihood_rates:

			if key is not None:
				continue
			# pull out both the bg counts and the pdfs, we need 'em both
			bgcounts_ba = self.background_likelihood_rates[key]
			bgpdf_ba = self.background_likelihood_pdfs[key]
			# we also need the zero lag counts to build the extinction model
			zlagcounts_ba = self.zero_lag_likelihood_rates[key]

			# Only model background above a ln(LR) of 3 or at the 10000 event count whatever is greater
			if zlagcounts_ba.array.sum() < 10000:
				likethreshvalue = 3.
				# Issue a warning if we have less than 10000 events
				warnings.warn("There are less than 10000 zerolag events, extinction effects on background may not be accurately calculated.")
			else:
				likethreshvalue = max(3, bgcounts_ba.bins.upper()[0][::-1][numpy.searchsorted(zlagcounts_ba.array[::-1].cumsum(), 10000)])
			likethreshindex = numpy.searchsorted(bgcounts_ba.bins.upper()[0], likethreshvalue)
			bgcounts_ba.array[:likethreshindex] = 0.
			bgpdf_ba.array[:likethreshindex] = 0.
			zlagcounts_ba.array[:likethreshindex] = 0.

			# safety checks
			assert not numpy.isnan(bgcounts_ba.array).any(), "log likelihood ratio rates contains NaNs"
			assert not (bgcounts_ba.array < 0.0).any(), "log likelihood ratio rate contains negative values"
			assert not numpy.isnan(bgpdf_ba.array).any(), "log likelihood ratio pdf contains NaNs"
			assert not (bgpdf_ba.array < 0.0).any(), "log likelihood ratio pdf contains negative values"

			# grab bins that are not infinite in size
			finite_bins = numpy.isfinite(bgcounts_ba.bins.volumes())
			ranks = bgcounts_ba.bins.upper()[0].compress(finite_bins)
			drank = bgcounts_ba.bins.volumes().compress(finite_bins)

			# figure out the minimum rank
			fit_min_rank = ranks[likethreshindex]

			# whittle down the arrays of counts and pdfs
			bgcounts_ba_array = bgcounts_ba.array.compress(finite_bins)
			bgpdf_ba_array = bgpdf_ba.array.compress(finite_bins)
			zlagcounts_ba_array = zlagcounts_ba.array.compress(finite_bins)

			def extinct(bgcounts_ba_array, bgpdf_ba_array, zlagcounts_ba_array, ranks, drank, fit_min_rank):
				# Generate arrays of complementary cumulative counts
				# for background events (monte carlo, pre clustering)
				# and zero lag events (observed, post clustering)
				zero_lag_compcumcount = zlagcounts_ba_array[::-1].cumsum()[::-1]
				bg_compcumcount = bgcounts_ba_array[::-1].cumsum()[::-1]

				# Fit for the number of preclustered, independent coincs by
				# only considering the observed counts safely in the bulk of
				# the distribution.  Only do the fit above 10 counts if there
				# are enough events.
				rank_range = numpy.logical_and(zero_lag_compcumcount <= max(zero_lag_compcumcount), zero_lag_compcumcount >= min(10, 0.001 * max(zero_lag_compcumcount)))

				# Use curve fit to find the predicted total preclustering
				# count. First we need an interpolator of the counts
				obs_counts = interpolate.interp1d(ranks, bg_compcumcount)
				bg_pdf_interp = interpolate.interp1d(ranks, bgpdf_ba_array)

				def extincted_counts(x, N_ratio, num_clustered = max(zero_lag_compcumcount), norm = obs_counts(fit_min_rank)):
					out = num_clustered * (1. - numpy.exp(-obs_counts(x) * N_ratio))
					# This normalization ensures that the left edge does go to num_clustered
					if (1. - numpy.exp(-norm * N_ratio)) > 0:
						out /= (1. - numpy.exp(-norm * N_ratio))
					out[~numpy.isfinite(out)] = 0.
					return out

				def extincted_pdf(x, N_ratio):
					out = numpy.exp(numpy.log(N_ratio) - obs_counts(x) * N_ratio + numpy.log(bg_pdf_interp(x)))
					out[~numpy.isfinite(out)] = 0.
					return out

				# Fit for the ratio of unclustered to clustered triggers.
				# Only fit N_ratio over the range of ranks decided above
				precluster_normalization, precluster_covariance_matrix = optimize.curve_fit(
					extincted_counts,
					ranks[rank_range],
					zero_lag_compcumcount.compress(rank_range),
					sigma = zero_lag_compcumcount.compress(rank_range)**.5,
					p0 = [3e-4, max(zero_lag_compcumcount)]
				)

				N_ratio, total_num = precluster_normalization

				return extincted_pdf(ranks, N_ratio)

			# get the extincted background PDF
			self.background_likelihood_rates[key].array[finite_bins] = extinct(bgcounts_ba_array, bgpdf_ba_array, zlagcounts_ba_array, ranks, drank, fit_min_rank) * drank
			self.zero_lag_likelihood_rates[key].array[:] = zlagcounts_ba.array[:]
		# Make sure the PDFs are all updated
		self.finish()

		# FIXME implement survival function
		return self, None

	@classmethod
	def from_xml(cls, xml, name):
		# find the root of the XML tree containing the
		# serialization of this object
		xml, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == u"%s:%s" % (name, cls.ligo_lw_name_suffix)]

		# create a mostly uninitialized instance
		self = cls(None, (), process_id = ligolw_param.get_pyvalue(xml, u"process_id"))

		# pull out the likelihood count and PDF arrays
		def reconstruct(xml, prefix, target_dict):
			for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and ("_%s" % prefix) in elem.Name]:
				ifo_set = frozenset(lsctables.instrumentsproperty.get(ba_elem.Name.split("_")[0]))
				target_dict[ifo_set] = rate.BinnedArray.from_xml(ba_elem, ba_elem.Name.split(":")[0])
		reconstruct(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		reconstruct(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		reconstruct(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		reconstruct(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)
		reconstruct(xml, u"zero_lag_likelihood_rate", self.zero_lag_likelihood_rates)
		reconstruct(xml, u"zero_lag_likelihood_pdf", self.zero_lag_likelihood_pdfs)

		if set(self.background_likelihood_pdfs):
			assert set(self.background_likelihood_rates) == set(self.background_likelihood_pdfs)
		if set(self.signal_likelihood_pdfs):
			assert set(self.signal_likelihood_rates) == set(self.signal_likelihood_pdfs)
		if set(self.zero_lag_likelihood_pdfs):
			assert set(self.zero_lag_likelihood_rates) == set(self.zero_lag_likelihood_pdfs)
		assert set(self.background_likelihood_rates) == set(self.signal_likelihood_rates)
		assert set(self.background_likelihood_rates) == set(self.zero_lag_likelihood_rates)

		self._compute_combined_rates()

		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"process_id", self.process_id))
		def store(xml, prefix, source_dict):
			for key, binnedarray in source_dict.items():
				if key is not None:
					ifostr = lsctables.instrumentsproperty.set(key).replace(",","")
					xml.appendChild(binnedarray.to_xml(u"%s_%s" % (ifostr, prefix)))
		store(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		store(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		store(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		store(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)
		store(xml, u"zero_lag_likelihood_rate", self.zero_lag_likelihood_rates)
		store(xml, u"zero_lag_likelihood_pdf", self.zero_lag_likelihood_pdfs)

		return xml


#
# Class to compute false-alarm probabilities and false-alarm rates from
# ranking statistic PDFs
#


class FAPFAR(object):
	def __init__(self, ranking_stats, livetime = None):
		# none is OK, but then can only compute FAPs, not FARs
		self.livetime = livetime

		# record trials factor, with safety checks
		counts = ranking_stats.zero_lag_likelihood_rates[None].array
		assert not numpy.isnan(counts).any(), "zero lag log likelihood ratio rates contain NaNs"
		assert (counts >= 0.).all(), "zero lag log likelihood ratio rates contain negative values"
		self.zero_lag_total_count = counts.sum()

		# get noise model ranking stat values and event counts from
		# bins
		ranks = ranking_stats.background_likelihood_rates[None].bins[0].upper()
		counts = ranking_stats.background_likelihood_rates[None].array
		assert not numpy.isnan(counts).any(), "background log likelihood ratio rates contain NaNs"
		assert (counts >= 0.).all(), "background log likelihood ratio rates contain negative values"

		# complementary cumulative distribution function
		ccdf = counts[::-1].cumsum()[::-1]
		ccdf /= ccdf[0]

		# ccdf is P(ranking stat > threshold | a candidate).  we
		# need P(ranking stat > threshold), i.e. need to correct
		# for the possibility that no candidate is present.
		# specifically, the ccdf needs to =1-1/e at the candidate
		# identification threshold, and cdf=1/e at the candidate
		# threshold, in order for FAR(threshold) * livetime to
		# equal the actual observed number of candidates.
		ccdf = gstlalstats.poisson_p_not_0(ccdf)

		# interpolator won't accept infinite co-ordinates so need
		# to remove the last bin
		ranks = ranks[:-1]
		ccdf = ccdf[:-1]

		# safety checks
		assert not numpy.isnan(ranks).any(), "log likelihood ratio co-ordinates contain NaNs"
		assert not numpy.isinf(ranks).any(), "log likelihood ratio co-ordinates are not all finite"
		assert not numpy.isnan(ccdf).any(), "log likelihood ratio CCDF contains NaNs"
		assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "log likelihood ratio CCDF failed to be normalized"

		# build interpolator.
		self.ccdf_interpolator = interpolate.interp1d(ranks, ccdf)

		# record min and max ranks so we know which end of the ccdf
		# to use when we're out of bounds
		self.minrank = ranks[0]
		self.maxrank = ranks[-1]

	@gstlalstats.assert_probability
	def ccdf_from_rank(self, rank):
		return self.ccdf_interpolator(numpy.clip(rank, self.minrank, self.maxrank))

	@gstlalstats.assert_probability
	def fap_from_rank(self, rank):
		# implements equation (8) from Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.
		return gstlalstats.fap_after_trials(self.ccdf_from_rank(rank), self.zero_lag_total_count)

	def rank_from_fap(self, p, tolerance = 1e-6):
		"""
		Inverts .fap_from_rank().  This function is sensitive to
		numerical noise for probabilities that are close to 1.  The
		tolerance sets the absolute error of the result.
		"""
		assert 0. <= p <= 1., "p (%g) is not a valid probability" % p
		lo, hi = self.minrank, self.maxrank
		while hi - lo > tolerance:
			mid = (hi + lo) / 2.
			mid_fap = self.fap_from_rank(mid)
			if p > mid_fap:
				# desired rank is below the middle
				hi = mid
			elif p < mid_fap:
				# desired rank is above the middle
				lo = mid
			else:
				# jackpot
				return mid
		return (hi + lo) / 2.
	rank_from_fap.__get__ = numpy.vectorize(rank_from_fap, otypes = (numpy.float64,), excluded = (0, 2, 'self', 'tolerance'))

	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.  the return value is divided by T to
		# convert events/experiment to events/second.  "tdp" =
		assert self.livetime is not None, "cannot compute FAR without livetime"
		# true-dismissal probability = 1 - single-event false-alarm
		# probability, the integral in equation (B4)
		log_tdp = numpy.log1p(-self.ccdf_from_rank(rank))
		return self.zero_lag_total_count * -log_tdp / self.livetime

	def rank_from_far(self, rate, tolerance = 1e-6):
		"""
		Inverts .far_from_rank() using a bisection search.  The
		tolerance sets the absolute error of the result.
		"""
		lo, hi = self.minrank, self.maxrank
		while hi - lo > tolerance:
			mid = (hi + lo) / 2.
			mid_far = self.far_from_rank(mid)
			if rate > mid_far:
				# desired rank is below the middle
				hi = mid
			elif rate < mid_far:
				# desired rank is above the middle
				lo = mid
			else:
				# jackpot
				return mid
		return (hi + lo) / 2.
	rank_from_far.__get__ = numpy.vectorize(rank_from_far, otypes = (numpy.float64,), excluded = (0, 2, 'self', 'tolerance'))

	def assign_fapfars(self, connection):
		# assign false-alarm probabilities and false-alarm rates
		# FIXME:  choose function names more likely to be unique?
		# FIXME:  abusing false_alarm_rate column to store FAP,
		# move to a false_alarm_probability column??
		def as_float(f):
			def g(x):
				return float(f(x))
			return g
		connection.create_function("fap", 1, as_float(self.fap_from_rank))
		connection.create_function("far", 1, as_float(self.far_from_rank))
		connection.cursor().execute("""
UPDATE
	coinc_inspiral
SET
	false_alarm_rate = (
		SELECT
			fap(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id
	),
	combined_far = (
		SELECT
			far(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id
	)
""")


#
# =============================================================================
#
#                                     I/O
#
# =============================================================================
#


def gen_likelihood_control_doc(xmldoc, process, coinc_params_distributions, ranking_data, seglists, name = u"gstlal_inspiral_likelihood", comment = None):
	node = xmldoc.childNodes[-1]
	assert node.tagName == ligolw.LIGO_LW.tagName

	if coinc_params_distributions is not None:
		coinc_params_distributions.process_id = process.process_id
		node.appendChild(coinc_params_distributions.to_xml(name))

	if ranking_data is not None:
		ranking_data.process_id = process.process_id
		node.appendChild(ranking_data.to_xml(name))

	llwsegments = ligolw_segments.LigolwSegments(xmldoc)
	llwsegments.insert_from_segmentlistdict(seglists, u"%s:segments" % name, comment = comment)
	llwsegments.finalize(process)

	return xmldoc


def parse_likelihood_control_doc(xmldoc, name = u"gstlal_inspiral_likelihood"):
	coinc_params_distributions = ranking_data = process_id = None
	try:
		coinc_params_distributions = ThincaCoincParamsDistributions.from_xml(xmldoc, name)
	except ValueError:
		pass
	else:
		process_id = coinc_params_distributions.process_id
	try:
		ranking_data = RankingData.from_xml(xmldoc, name)
	except ValueError:
		pass
	else:
		if process_id is None:
			process_id = ranking_data.process_id
	if coinc_params_distributions is None and ranking_data is None:
		raise ValueError("document does not contain likelihood ratio data")
	seglists = ligolw_segments.segmenttable_get_by_name(xmldoc, u"%s:segments" % name).coalesce()
	return coinc_params_distributions, ranking_data, seglists


def marginalize_pdf_urls(urls, require_coinc_param_data, require_ranking_stat_data, ignore_missing_files = False, verbose = False):
	"""
	Implements marginalization of PDFs in ranking statistic data files.
	The marginalization is over the degree of freedom represented by
	the file collection.  One or both of the candidate parameter PDFs
	and ranking statistic PDFs can be processed, with errors thrown if
	one or more files is missing the required component.
	"""
	distributions = None
	ranking_data = None
	seglists = segments.segmentlistdict()
	for n, url in enumerate(urls, start = 1):
		#
		# load input document
		#

		if verbose:
			print >>sys.stderr, "%d/%d:" % (n, len(urls)),
		try:
			in_xmldoc = ligolw_utils.load_url(url, verbose = verbose, contenthandler = ThincaCoincParamsDistributions.LIGOLWContentHandler)
		except IOError:
			# IOError is raised when an on-disk file is
			# missing.  urllib2.URLError is raised when a URL
			# cannot be loaded, but this is subclassed from
			# IOError so IOError will catch those, too.
			if not ignore_missing_files:
				raise
			if verbose:
				print >>sys.stderr, "Could not load \"%s\" ... skipping as requested" % url
			continue

		#
		# compute weighted sum of ranking data PDFs
		#

		this_distributions, this_ranking_data, this_seglists = parse_likelihood_control_doc(in_xmldoc)
		in_xmldoc.unlink()

		if this_distributions is None and require_coinc_param_data:
			raise ValueError("\"%s\" contains no parameter PDF data" % url)
		if this_ranking_data is None and require_ranking_stat_data:
			raise ValueError("\"%s\" contains no ranking statistic PDF data" % url)

		if distributions is None:
			distributions = this_distributions
		elif this_distributions is not None:
			distributions += this_distributions
		if ranking_data is None:
			ranking_data = this_ranking_data
		elif this_ranking_data is not None:
			ranking_data += this_ranking_data
		seglists |= this_seglists

	if distributions is None and ranking_data is None:
		raise ValueError("no data loaded from input documents")

	return distributions, ranking_data, seglists


#
# =============================================================================
#
#                                    Other
#
# =============================================================================
#


def get_live_time(seglistdict, min_instruments = 2, verbose = False):
	if min_instruments < 1:
		raise ValueError("min_instruments (=%d) must be >= 1" % min_instruments)
	livetime = float(abs(vote((segs for instrument, segs in seglistdict.items() if instrument != "H2"), min_instruments)))
	if verbose:
		print >> sys.stderr, "Livetime: %.3g s" % livetime
	return livetime
