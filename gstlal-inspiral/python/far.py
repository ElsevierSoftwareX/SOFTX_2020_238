# Copyright (C) 2011--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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


import copy
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
from scipy import interpolate
from scipy import optimize
from scipy import stats
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
sqlite3.enable_callback_tracebacks(True)
import sys


from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments
from glue.segmentsUtils import vote
from gstlal import emcee
from pylal import inject
from pylal import progress
from pylal import rate
from pylal import snglcoinc


#
# =============================================================================
#
#                                Binomial Stuff
#
# =============================================================================
#


def fap_after_trials(p, m):
	"""
	Given the probability, p, that an event occurs, compute the
	probability of at least one such event occuring after m independent
	trials.

	The return value is 1 - (1 - p)^m computed avoiding round-off
	errors and underflows.  m cannot be negative but need not be an
	integer.
	"""
	# 1 - (1 - p)^m = m p - (m^2 - m) p^2 / 2 +
	#	(m^3 - 3 m^2 + 2 m) p^3 / 6 -
	#	(m^4 - 6 m^3 + 11 m^2 - 6 m) p^4 / 24 + ...
	#
	# starting at 0, the nth term in the series is
	#
	# -1^n * (m - 0) * (m - 1) * ... * (m - n) * p^(n + 1) / (n + 1)!
	#
	# if the (n-1)th term is X, the nth term in the series is
	#
	# X * (n - m) * p / (n + 1)
	#
	# which allows us to avoid explicit evaluation of each term's
	# numerator and denominator separately (each of which quickly
	# overflow).
	#
	# for sufficiently large n the denominator dominates and the terms
	# in the series become small and the sum eventually converges to a
	# stable value (and if m is an integer the sum is exact in a finite
	# number of terms).  however, if m*p >> 1 it can take many terms
	# before the series sum stabilizes, terms in the series initially
	# grow large and alternate in sign and an accurate result can only
	# be obtained through careful cancellation of the large values.
	#
	# for large m*p we take a different approach to evaluating the
	# result.  in this regime the result will be close to 1 so (1 -
	# p)^m will be small
	#
	# (1 - p)^m = exp(m ln(1 - p))
	#
	# if p is small, ln(1 - p) suffers from loss of precision but the
	# Taylor expansion of ln(1 - p) converges quickly
	#
	# m ln(1 - p) = -m p - m p^2 / 2 - m p^3 / 3 - ...
	#             = -m p * (1 + p / 2 + p^2 / 3 + ...)
	#
	# as an alternative, the standard library provides log1p(),
	# which evalutes ln(1 + p) accurately for small p.
	#
	# if p is close to 1, ln(1 - p) suffers a domain error
	#

	assert m >= 0, "m = %g cannot be negative" % m
	assert 0 <= p <= 1, "p = %g must be between 0 and 1 inclusively" % p

	if m * p < 1.0:
		#
		# use direct Taylor expansion of 1 - (1 - p)^m
		#

		s = 0.0
		term = -1.0
		for n in itertools.count():
			term *= (n - m) * p / (n + 1.0)
			s += term
			if abs(term) <= abs(1e-17 * s):
				return s

	if p < .125:
		#
		# compute result from Taylor expansion of ln(1 - p)
		#

		return 1.0 - math.exp(m * math.log1p(-p))

		#
		# original implementation in case log1p() gives us problems
		#

		s = p_powers = 1.0
		for n in itertools.count(2):
			p_powers *= p
			term = p_powers / n
			s += term
			if term <= 1e-17 * s:
				return 1.0 - math.exp(-m * p * s)

	try:
		#
		# try direct evaluation of 1 - exp(m ln(1 - p))
		#

		return 1.0 - math.exp(m * math.log(1.0 - p))

	except ValueError:
		#
		# math.log has suffered a domain error, therefore p is very
		# close to 1.  we know p <= 1 because it's a probability,
		# and we know that m*p >= 1 otherwise we wouldn't have
		# followed this code path, therefore m >= 1, and so because
		# p is close to 1 and m is not small we can safely assume
		# the anwer is 1.
		#

		return 1.0


def trials_from_faps(p0, p1):
	"""
	Given the probabiity, p0, of an event occuring, and the
	probability, p1, of at least one such event being observed after
	some number of independent trials, solve for and return the number
	of trials, m, that relates the two probabilities.  The three
	quantities are related by p1 = 1 - (1 - p0)^m.  Generally the
	return value is not an integer.

	See also fap_after_trials().  Note that if p0 is 0 or 1 then p1
	must be 0 or 1 respectively, and in both cases m is undefined.
	Otherwise if p1 is 1 then inf is returned.
	"""
	assert 0 <= p0 <= 1	# p0 must be a valid probability
	assert 0 <= p1 <= 1	# p1 must be a valid probability

	if p0 == 0 or p0 == 1:
		assert p0 == p1	# require valid relationship
		# but we still can't solve for m
		raise ValueError("m undefined")
	if p1 == 1:
		return PosInf

	#
	# find range of m that contains solution.  the false alarm
	# probability increases monotonically with the number of trials
	#

	lo, hi = 0, 100
	while fap_after_trials(p0, hi) < p1:
		hi *= 100

	#
	# use fap_after_trials() and scipy's Brent root finder to solve for
	# m
	#

	return optimize.brentq((lambda m: fap_after_trials(p0, m) - p1), lo, hi)


#
# =============================================================================
#
#                 Parameter Distributions Book-Keeping Object
#
# =============================================================================
#


#
# FAR normalization helper
#


class CountAboveThreshold(dict):
	"""
	Device for counting the number of zero-lag coincs above threshold
	as a function of the instruments that participated.
	"""
	def update(self, connection, coinc_def_id, threshold):
		for instruments, count in connection.cursor().execute("""
SELECT
	coinc_inspiral.ifos,
	COUNT(*)
FROM
	coinc_inspiral
	JOIN coinc_event ON (
		coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id
	)
WHERE
	coinc_event.coinc_def_id == ?
	AND coinc_event.likelihood >= ?
	AND NOT EXISTS (
		SELECT
			*
		FROM
			time_slide
		WHERE
			time_slide.time_slide_id == coinc_event.time_slide_id
			AND time_slide.offset != 0
	)
GROUP BY
	coinc_inspiral.ifos
""", (coinc_def_id, threshold)):
			try:
				self[frozenset(lsctables.instrument_set_from_ifos(instruments))] += count
			except KeyError:
				self[frozenset(lsctables.instrument_set_from_ifos(instruments))] = count


#
# Inspiral-specific CoincParamsDistributions sub-class
#


class ThincaCoincParamsDistributions(snglcoinc.CoincParamsDistributions):
	# FIXME:  switch to new default when possible
	ligo_lw_name_suffix = u"pylal_ligolw_burca_tailor_coincparamsdistributions"

	instrument_categories = snglcoinc.InstrumentCategories()

	# range of SNRs covered by this object
	# FIXME:  must ensure lower boundary matches search threshold
	snr_min = 4.
	snr_max = 100.

	# binnings and filters
	binnings = {
		"instruments": rate.NDBins((rate.LinearBins(0.5, instrument_categories.max() + 0.5, instrument_categories.max()),)),
		"H1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(snr_min, snr_max, 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"H2_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(snr_min, snr_max, 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"H1H2_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(snr_min, snr_max, 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"L1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(snr_min, snr_max, 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"V1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(snr_min, snr_max, 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200)))
	}

	def snr_chi_filter(bins, snr_width_at_8 = math.sqrt(2) / 4.0, sigma = 10):
		snr_bins = bins[0]
		bin_width_at_8 = (snr_bins.upper() - snr_bins.lower())[snr_bins[8.0]]
		return rate.gaussian_window(snr_width_at_8 / bin_width_at_8, 7, sigma = sigma)

	filters = {
		"H1_snr_chi": snr_chi_filter(binnings["H1_snr_chi"]),
		"H2_snr_chi": snr_chi_filter(binnings["H2_snr_chi"]),
		"H1H2_snr_chi": snr_chi_filter(binnings["H1H2_snr_chi"]),
		"L1_snr_chi": snr_chi_filter(binnings["L1_snr_chi"]),
		"V1_snr_chi": snr_chi_filter(binnings["V1_snr_chi"])
	}

	del snr_chi_filter

	#
	# class-level cache of pre-computed SNR joint PDFs.  structure is like
	#
	# 	= {
	#		frozenset([("H1", horiz_dist), ("L1", horiz_dist)]): (InterpBinnedArray, BinnedArray, age),
	#		...
	#	}
	#
	# at most max_cached_snr_joint_pdfs will be stored.  if a PDF is
	# required that cannot be found in the cache, and there is no more
	# room, the oldest PDF (the one with the *smallest* age) is pop()ed
	# to make room, and the new one added with its age set to the
	# highest age in the cache + 1.
	#

	max_cached_snr_joint_pdfs = 15
	snr_joint_pdf_cache = {}

	def get_snr_joint_pdf(self, instrument_horizon_distance_mapping, log_distance_tolerance = math.log(1.2)):
		#
		# key for cache:  frozen set of (instrument, horizon
		# distance) pairs.  horizon distances are normalized to
		# fractions of the largest among them and then are
		# quantized to integer powers of
		# exp(log_distance_tolerance)
		#
		# FIXME:  is the default distance tolerance appropriate?
		#
		# FIXME:  if horizon distance discrepancy is too large,
		# consider a fast-path that just returns an all-0 array
		#

		horizon_distance_norm = max(instrument_horizon_distance_mapping.values())
		key = frozenset((instrument, math.exp(math.floor(math.log(horizon_distance / horizon_distance_norm) / log_distance_tolerance) * log_distance_tolerance)) for instrument, horizon_distance in instrument_horizon_distance_mapping.items())

		#
		# retrieve cached PDF, or build new one
		#

		try:
			# FIXME:  need to check that the result is
			# appropriate for the SNR threshold and regenerate
			# if not
			pdf = self.snr_joint_pdf_cache[key][0]
		except KeyError:
			# no entries in cache for this instrument combo and
			# set of horizon distances
			if self.snr_joint_pdf_cache:
				age = max(age for ignored, ignored, age in self.snr_joint_pdf_cache.values()) + 1
			else:
				age = 0
			# FIXME:  turn off verbose
			binnedarray = joint_pdf_of_snrs(dict(key), snr_threshold = self.snr_min, snr_max = self.snr_max, verbose = True)
			pdf = rate.InterpBinnedArray(binnedarray)
			self.snr_joint_pdf_cache[key] = pdf, binnedarray, age
			# if the cache is full, delete the entry with the
			# smallest age
			while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
				del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
		return pdf

	@staticmethod
	def coinc_params(events, offsetvector):
		# FIXME:  extract horizon distances from sngl_inspiral
		# triggers and add an instrument-->horizon distance mapping
		# to the params dictionary
		params = dict(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events)
		# don't allow both H1 and H2 to participate in the same
		# coinc.  if both have participated favour H1
		if "H2_snr_chi" in params and "H1_snr_chi" in params:
			del params["H2_snr_chi"]
		params["instruments"] = (ThincaCoincParamsDistributions.instrument_categories.category(event.ifo for event in events),)
		return params

	def P_signal(self, params):
		if params is None:
			return None
		# (instrument, snr) pairs sorted alphabetically by instrument name
		snrs = sorted((name.split("_")[0], value[0]) for name, value in params.items() if name.endswith("_snr_chi"))
		# retrieve the SNR PDF
		# FIXME:  get instrument-->horizon distance mapping from
		# params
		snr_pdf = self.get_snr_joint_pdf(dict((instrument, self.horizon_distances[instrument]) for instrument, rho in snrs))
		# evaluate it (snrs are alphabetical by instrument)
		P = snr_pdf(*tuple(rho for instrument, rho in snrs))

		# FIXME:  P(instruments | signal) needs to depend on
		# horizon distances.  here we're assuming whatever
		# add_foreground_prior() has set the probabilities to is
		# OK.  we probably need to cache these and save them in the
		# XML file, too, like P(snrs | signal, instruments)
		for name, value in params.items():
			P *= self.injection_pdf_interp[name](*value)
		return P

	@staticmethod
	def create_emcee_lnprob_wrapper(lnprobfunc, keys):
		keys = tuple(sorted(keys))
		def coinc_params_from_flat_args(coords):
			# coords[0::2] = rho
			# coords[1::2] = chi^2/rho^2
			params = dict(zip(keys, zip(coords[0::2], coords[1::2])))
			# FIXME:  add instruments when needed
			return params
		return lambda coords: lnprobfunc(coinc_params_from_flat_args(coords))

	def add_background_prior(self, segs, n = 1., transition = 10., prefactors_range = (1.0, 10.0), df = 40, verbose = False):
		#
		# populate instrument combination binning.  assume the
		# single-instrument categories in the background rates
		# instruments binning provide the background event counts
		# for the segment lists provided.  NOTE:  we're using the
		# counts in the single-instrument categories for input, and
		# the output only goes into the 2-instrument and higher
		# categories, so this procedure does not result in a loss
		# of information and can be performed multiple times
		# without altering the statistics of the input data.
		#
		# FIXME:  we need to know the coincidence window to do this
		# correctly.  we assume 5ms.  get the correct number.
		#
		# FIXME:  should this be done in .finish()?  but we'd need
		# the segment lists

		if verbose:
			print >>sys.stderr, "synthesizing background-like instrument combination probabilities..."
		coincsynth = snglcoinc.CoincSynthesizer(
			eventlists = dict((instrument, self.background_rates["instruments"][self.instrument_categories.category([instrument]),]) for instrument in segs),
			segmentlists = segs,
			delta_t = 0.005
		)
		# assume the single-instrument events are being collected
		# in several disjoint bins so that events from different
		# instruments that occur at the same time but in different
		# bins are not coincident.  if there are M bins for each
		# instrument, the probability that N events all occur in
		# the same bin is (1/M)^(N-1).  the number of bins, M, is
		# therefore given by the (N-1)th root of the ratio of the
		# predicted number of N-instrument coincs to the observed
		# number of N-instrument coincs.  use the average of M
		# measured from all instrument combinations.
		#
		# finding M by comparing predicted to observed zero-lag
		# counts assumes we are in a noise-dominated regime, i.e.,
		# that the observed relative abundances of coincs are not
		# significantly contaminated by signals.  if signals are
		# present in large numbers, and occur in different
		# abundances than the noise events, averaging the apparent
		# M over different instrument combinations helps to
		# suppress the contamination.  NOTE:  the number of
		# coincidence bins, M, should be very close to the number
		# of templates (experience shows that it is not equal to
		# the number of templates, though I don't know why).
		livetime = get_live_time(segs)
		n = 0
		coincidence_bins = 0.
		for instruments in coincsynth.all_instrument_combos:
			predicted_count = coincsynth.rates[frozenset(instruments)] * livetime
			observed_count = self.zero_lag_rates["instruments"][self.instrument_categories.category(instruments),]
			if predicted_count > 0 and observed_count > 0:
				coincidence_bins += (predicted_count / observed_count)**(1. / (len(instruments) - 1))
				n += 1
		coincidence_bins /= n
		if verbose:
			print >>sys.stderr, "\tthere seems to be %g effective disjoint coincidence bin(s)" % coincidence_bins
		if math.isnan(coincidence_bins) or coincidence_bins == 0.:
			# in these cases all the rates are just 0
			for instruments in coincsynth.all_instrument_combos:
				self.background_rates["instruments"][self.instrument_categories.category(instruments),] = 0.
		else:
			assert coincidence_bins >= 1.
			# convert single-instrument event rates to rates/bin
			coincsynth.mu = dict((instrument, rate / coincidence_bins) for instrument, rate in coincsynth.mu.items())
			# now compute the expected coincidence rates/bin,
			# then multiply by the number of bins to get the
			# expected coincidence rates
			for instruments, count in coincsynth.mean_coinc_count.items():
				self.background_rates["instruments"][self.instrument_categories.category(instruments),] = count * coincidence_bins

		#
		# populate snr,chi2 binnings
		#

		if verbose:
			print >>sys.stderr, "synthesizing background-like (SNR, \\chi^2) distributions..."
		for instrument in segs:
			binarr = self.background_rates["%s_snr_chi" % instrument]
			if verbose:
				progressbar = progress.ProgressBar(instrument, max = len(binarr.bins[0]))
			else:
				progressbar = None

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)
			# Custom handle the first and last over flow bins
			snrs = new_binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = new_binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for snr in snrs:
				p = math.exp(-snr**2 / 2. + snrs[0]**2 / 2. + math.log(n))
				p += (transition / snr)**6 * math.exp(-transition**2 / 2. + snrs[0]**2 / 2. + math.log(n)) # Softer fall off above some transition SNR for numerical reasons
				for chi2_over_snr2 in chi2_over_snr2s:
					new_binarr[snr, chi2_over_snr2] += p
				if progressbar is not None:
					progressbar.increment()
			# normalize to the requested count
			new_binarr.array *= n / new_binarr.array.sum()
			# add to raw counts
			binarr += new_binarr

		# FIXME, an adhoc way of adding glitches, use a signal distribution with bad matches
		self.add_foreground_snrchi_prior(self.background_rates, instruments = set(segs), n = n, prefactors_range = prefactors_range, df = df, verbose = verbose)

	def add_foreground_snrchi_prior(self, target_dict, instruments, n, prefactors_range, df, verbose = False):
		if verbose:
			print >>sys.stderr, "synthesizing signal-like (SNR, \\chi^2) distributions..."
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 10)
		for instrument in instruments:
			binarr = target_dict["%s_snr_chi" % instrument]
			if verbose:
				progressbar = progress.ProgressBar(instrument, max = len(binarr.bins[0]))
			else:
				progressbar = None

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)

			# Custom handle the first and last over flow bins
			snrs = new_binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = new_binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for snr in snrs:
				for chi2_over_snr2 in chi2_over_snr2s:
					chisq = chi2_over_snr2 * snr**2 * df # We record the reduced chi2
					dist = 0
					for pf in pfs:
						nc = pf * snr**2
						v = stats.ncx2.pdf(chisq, df, nc)
						if numpy.isfinite(v):
							dist += v
					dist *= (snr / snrs[0])**-4
					if numpy.isfinite(dist):
						new_binarr[snr, chi2_over_snr2] += dist
				if progressbar is not None:
					progressbar.increment()
			# normalize to the requested count
			new_binarr.array *= n / new_binarr.array.sum()
			# add to raw counts
			binarr += new_binarr

	def add_foreground_prior(self, segs, n = 1., prefactors_range = (0.0, 0.10), df = 40, verbose = False):
		#
		# populate instrument combination binning
		#

		assert len(segs) > 1
		assert set(self.horizon_distances) <= set(segs)

		# probability that a signal is detectable by each of the
		# instrument combinations
		P = P_instruments_given_signal(self.horizon_distances, snr_threshold = self.snr_min)
		# multiply by probability that enough instruments are on to
		# form each of those combinations
		P_live = snglcoinc.CoincSynthesizer(segmentlists = segs).P_live
		for instruments in P:
			P[instruments] *= sum(sorted(p for on_instruments, p in P_live.items() if on_instruments >= instruments))
		# renormalize
		total = sum(sorted(P.values()))
		for instruments in P:
			P[instruments] /= total
		# populate binning from probabilities
		for instruments, p in P.items():
			self.injection_rates["instruments"][self.instrument_categories.category(instruments),] += n * p

		#
		# populate snr,chi2 binnings
		#

		self.add_foreground_snrchi_prior(self.injection_rates, instruments = set(segs), n = n, prefactors_range = prefactors_range, df = df, verbose = verbose)

	def _rebuild_interpolators(self):
		super(ThincaCoincParamsDistributions, self)._rebuild_interpolators()

		#
		# the instrument combination "interpolators" are pass-throughs
		#

		self.background_pdf_interp["instruments"] = lambda x: self.background_pdf["instruments"][x,]
		self.injection_pdf_interp["instruments"] = lambda x: self.injection_pdf["instruments"][x,]
		self.zero_lag_pdf_interp["instruments"] = lambda x: self.zero_lag_pdf["instruments"][x,]

	def finish(self, *args, **kwargs):
		super(ThincaCoincParamsDistributions, self).finish(*args, **kwargs)

		# NOTE:  because we use custom PDF constructions, the stock
		# .__iadd__() method for this class will not result in
		# valid PDFs.  the rates arrays *are* handled correctly by
		# the .__iadd__() method, by fiat, so just remember to
		# invoke .finish() to get the PDFs in shape afterwards

		# convert signal (aka injection) (rho, chi^2/rho^2) PDFs
		# into P(chi^2/rho^2 | rho)
		for name, pdf in self.injection_pdf.items():
			if not name.endswith("_snr_chi"):
				continue
			bin_sizes = pdf.bins[1].upper() - pdf.bins[1].lower()
			for i in xrange(pdf.array.shape[0]):
				nonzero = pdf.array[i] != 0
				pdf.array[i] /= numpy.dot(numpy.compress(nonzero, pdf.array[i]), numpy.compress(nonzero, bin_sizes))

		# instrument combos are probabilities, not densities.  be
		# sure the single-instrument categories are zeroed.
		self.background_pdf["instruments"] = self.background_rates["instruments"].copy()
		self.injection_pdf["instruments"] = self.injection_rates["instruments"].copy()
		self.zero_lag_pdf["instruments"] = self.zero_lag_rates["instruments"].copy()
		for category in self.instrument_categories.values():
			self.background_pdf["instruments"][category,] = 0
			self.injection_pdf["instruments"][category,] = 0
			self.zero_lag_pdf["instruments"][category,] = 0
		self.background_pdf["instruments"].array /= self.background_pdf["instruments"].array.sum()
		self.injection_pdf["instruments"].array /= self.injection_pdf["instruments"].array.sum()
		self.zero_lag_pdf["instruments"].array /= self.zero_lag_pdf["instruments"].array.sum()

		self._rebuild_interpolators()

	@classmethod
	def from_xml(cls, xml, name):
		self = super(ThincaCoincParamsDistributions, cls).from_xml(xml, name)
		xml = self.get_xml_root(xml, name)
		prefix = u"cached_snr_joint_pdf"
		for elem in [elem for elem in xml.childNodes if elem.getAttribute(u"Name").startswith(u"%s:" % prefix)]:
			key = frozenset((inst.strip(), float(dist.strip())) for inst, dist in (inst_dist.strip().split(u"=") for inst_dist in ligolw_param.get_pyvalue(elem, u"key").strip().split(u",")))
			binnedarray = rate.binned_array_from_xml(elem, prefix)
			if self.snr_joint_pdf_cache:
				age = max(age for ignored, ignored, age in self.snr_joint_pdf_cache.values()) + 1
			else:
				age = 0
			self.snr_joint_pdf_cache[key] = rate.InterpBinnedArray(binnedarray), binnedarray, age
			while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
				del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
		return self

	def to_xml(self, name):
		xml = super(ThincaCoincParamsDistributions, self).to_xml(name)
		prefix = u"cached_snr_joint_pdf"
		for key, (ignored, binnedarray, ignored) in self.snr_joint_pdf_cache.items():
			elem = xml.appendChild(rate.binned_array_to_xml(binnedarray, prefix))
			elem.appendChild(ligolw_param.new_param(u"key", u"lstring", u",".join(u"%s=%.17g" % inst_dist for inst_dist in sorted(key))))
		return xml

	@property
	def count_above_threshold(self):
		"""
		Dictionary mapping instrument combination (as a frozenset)
		to number of zero-lag coincs observed.  An additional entry
		with key None stores the total.
		"""
		count_above_threshold = CountAboveThreshold((frozenset(self.instrument_categories.instruments(int(round(category)))), count) for category, count in zip(self.zero_lag_rates["instruments"].bins.centres()[0], self.zero_lag_rates["instruments"].array))
		count_above_threshold[None] = sum(sorted(count_above_threshold.values()))
		return count_above_threshold

	@count_above_threshold.setter
	def count_above_threshold(self, count_above_threshold):
		self.zero_lag_rates["instruments"].array[:] = 0.
		for instruments, count in count_above_threshold.items():
			if instruments is not None:
				self.zero_lag_rates["instruments"][self.instrument_categories.category(instruments),] = count

	@property
	def Pinstrument_noise(self):
		P = {}
		for category, p in zip(self.background_pdf["instruments"].bins.centres()[0], self.background_pdf["instruments"].array):
			instruments = frozenset(self.instrument_categories.instruments(int(round(category))))
			if len(instruments) < 2 or not p:
				continue
			P[instruments] = p
		return P

	@property
	def Pinstrument_signal(self):
		P = {}
		for category, p in zip(self.injection_pdf["instruments"].bins.centres()[0], self.injection_pdf["instruments"].array):
			instruments = frozenset(self.instrument_categories.instruments(int(round(category))))
			if len(instruments) < 2 or not p:
				continue
			P[instruments] = p
		return P

	@staticmethod
	def randindex(lo, hi, n = 1.):
		"""
		Yields integers in the range [lo, hi) where both lo and hi
		are not negative.  Each return value is a two-element
		tuple.  The first element is the random integer, the second
		is the natural logarithm of the probability with which that
		integer will be chosen.

		The CDF for the distribution from which the integers are
		drawn goes as [integer]^{n}.  Specifically, it's

			CDF(x) = (x^{n} - lo^{n}) / (hi^{n} - lo^{n})

		n = 1 yields a uniform distribution;  n > 1 favours
		larger integers, n < 1 favours smaller integers.
		"""
		# NOTE:  nothing requires the probabilities returned by
		# this generator to be properly normalized, but it turns
		# out to be trivial to achieve so we do it anyway, just in
		# case it turns out to be helpful later.

		if not 0 <= lo < hi:
			raise ValueError("require 0 <= lo < hi: lo = %d, hi = %d" % (lo, hi))
		if n < 0.:
			raise ValueError("n < 0: %g" % n)
		elif n == 0.:
			# special case for degenerate PDF
			while 1:
				yield lo, 0.
		elif n == 1.:
			# special case for uniform distribution
			lnP = math.log(1. / (hi - lo))
			hi -= 1
			rnd = random.randint
			while 1:
				yield rnd(lo, hi), lnP

		# CDF evaluated at index boundaries
		lnP = numpy.arange(lo, hi + 1, dtype = "double")**n
		lnP -= lnP[0]
		lnP /= lnP[-1]
		# differences give probabilities
		lnP = tuple(numpy.log(lnP[1:] - lnP[:-1]))

		beta = lo**n / (hi**n - lo**n)
		n = 1. / n
		alpha = hi / (1. + beta)**n
		flr = math.floor
		rnd = random.random
		while 1:
			index = int(flr(alpha * (rnd() + beta)**n))
			# the tuple look-up also provides the range safety
			# check on index
			yield index, lnP[index - lo]

	def random_params(self, instruments):
		"""
		Generator that yields an endless sequence of randomly
		generated parameter dictionaries for the given keys.  NOTE:
		the parameters will be within the domain of the repsective
		binnings but are not drawn from their PDF.  The return value is
		a tuple, the first element of which is the parameter dictionary
		and the second is the natural logarithm (up to an arbitrary
		constant) of the PDF from which the parameters have been drawn.
		"""
		snr_slope = 0.5

		keys = tuple("%s_snr_chi" % instrument for instrument in instruments)
		base_params = {"instruments": (self.instrument_categories.category(instruments),)}
		x = dict((key, self.binnings[key].centres()) for key in keys)
		ln_dxes = dict((key, tuple(numpy.log(u - l) for u, l in zip(self.binnings[key].upper(), self.binnings[key].lower()))) for key in keys)
		indexgen = tuple((key, (self.randindex(0, len(centres_a), snr_slope).next, self.randindex(0, len(centres_b)).next)) for key, (centres_a, centres_b) in x.items())
		isinf = math.isinf
		isnan = math.isnan
		while 1:
			indexes = tuple((key, (indexgen_a(), indexgen_b())) for key, (indexgen_a, indexgen_b) in indexgen)
			# P(index) / (size of bin) = probability density
			#
			# NOTE:  I think the result of this sum is, in
			# fact, correctly normalized, but nothing requires
			# it to be and I've not checked that it is so the
			# documentation doesn't promise that it is.
			ln_P = sum(sum(ln_P_i - ln_dx[i] for ln_dx, (i, ln_P_i) in zip(ln_dxes[key], value)) for key, value in indexes)
			if not (isinf(ln_P) or isnan(ln_P)):
				params = dict((key, tuple(centres[i] for centres, (i, ln_P_i) in zip(x[key], value))) for key, value in indexes)
				params.update(base_params)
				yield params, ln_P


#
# Joint probability density for measured SNRs
#


def joint_pdf_of_snrs(inst_horiz_mapping, snr_threshold, snr_max, n_samples = 10000, decades_per_bin = 1.0 / 50.0, verbose = False):
	"""
	A function which returns a BinnedArray representing the joint
	probability density of measuring a set of SNRs from a network of
	instruments.  The inst_horiz_mapping is a dictionary mapping
	instrument name (e.g., "H1") to horizon distance (arbitrary units).
	snr_threshold is the lowest accepted SNR (must be > 3), and
	n_samples is the number of lines over which to calculate the
	density in the SNR space.  The axes of the PDF correspond to the
	instruments in alphabetical order.
	"""
	snr_threshold = float(snr_threshold)	# just to be sure
	snr_max = float(snr_max)	# just to be sure

	# An effective threshold used in the calculations in order to simulate
	# noise effects
	snr_min = snr_threshold - 3.0
	assert snr_max > snr_threshold
	assert snr_min > 0.0

	# get instrument names in alphabetical order
	names = sorted(inst_horiz_mapping)
	# get horizon distances and responses in that same order
	DH = numpy.array([inst_horiz_mapping[inst] for inst in names])
	resps = [inject.cached_detector[inject.prefix_to_name[inst]].response for inst in names]

	pdf = rate.BinnedArray(rate.NDBins([rate.LogarithmicBins(snr_min, snr_max, int(round(math.log10(snr_max / snr_min) / decades_per_bin)))] * len(names)))

	steps_per_bin = 3.
	decades_per_step = decades_per_bin / steps_per_bin
	_per_step = 10.**decades_per_step - 1.

	psi = gmst = 0.0

	if verbose:
		progressbar = progress.ProgressBar("%s SNR joint PDF" % ", ".join(names), max = n_samples)
	else:
		progressbar = None

	for i in xrange(n_samples):
		theta = math.acos(random.uniform(-1., 1.))
		phi = random.uniform(0., 2. * math.pi)
		cosi2 = random.uniform(-1., 1.)**2.

		fpfc2 = numpy.array([inject.XLALComputeDetAMResponse(resp, phi, math.pi / 2. - theta, psi, gmst) for resp in resps])**2.

		# ratio of inverse SNR to distance for each instrument
		snr_times_D = 8. * DH * numpy.dot(fpfc2, numpy.array([(1. + cosi2)**2. / 4., cosi2]))**0.5

		# index of instrument whose SNR grows fastest with decreasing D
		axis = snr_times_D.argmax()

		# furthest an event can be and still be above snr_min in
		# all instruments, and the SNR that corresponds to in the
		# instrument whose SNR grows fastest
		snr_start = snr_times_D[axis] * (snr_min / snr_times_D.min())

		# 3 steps per bin
		for snr in 10.**numpy.arange(math.log10(snr_start), math.log10(snr_max), decades_per_step):
			# "snr" is SNR in fastest growing instrument, from
			# this the distance to the source is:
			#
			#	D = snr_times_D[axis] / snr
			#
			# and the SNRs in all instruments are:
			#
			#	snr_times_D / D
			#
			# but round-off protection is required to ensure
			# all SNRs are within the allowed range
			#
			# SNR step size:
			#	d(snr) = (10**decades_per_step - 1.) * snr
			#
			# rate of change of D with SNR:
			#	dD/d(snr) = -snr_times_D / snr^2
			#	          = -D / snr
			#
			# relationship b/w dD and d(snr):
			#	dD = -D / snr d(snr)
			#	   = -D * (10**decades_per_step - 1.)
			#
			# number of sources:
			#	\propto D^2 |dD|
			#	\propto D^3 * (10**decades_per_step - 1.)
			D = snr_times_D[axis] / snr
			pdf[tuple((snr_times_D / D).clip(snr_min, PosInf))] += D**3. * _per_step

		if progressbar is not None:
			progressbar.increment()

	# number of bins per unit in SNR in the binnings.  For use as the
	# width parameter in the filtering.
	bins_per_snr_at_8 = 1. / ((10.**decades_per_bin - 1.) * 8.)
	rate.filter_array(pdf.array,rate.gaussian_window(*([math.sqrt(2.) * bins_per_snr_at_8] * len(inst_horiz_mapping))))
	numpy.clip(pdf.array, 0, PosInf, pdf.array)
	# set the region where any SNR is lower than the input threshold to
	# zero before normalizing the pdf and returning.
	range_all = slice(None,None)
	range_low = slice(snr_min, snr_threshold)
	for i in xrange(len(inst_horiz_mapping)):
		slices = [range_all] * len(inst_horiz_mapping)
		slices[i] = range_low
		pdf[tuple(slices)] = 0
	pdf.to_pdf()
	return pdf


def P_instruments_given_signal(inst_horiz_mapping, snr_threshold, n_samples = 500000):
	# FIXME:  this function does not yet incorporate the effect of
	# noise-induced SNR fluctuations in its calculations

	# get instrument names
	names = tuple(inst_horiz_mapping)
	# get horizon distances and responses in that same order
	DH = numpy.array([inst_horiz_mapping[inst] for inst in names])
	resps = [inject.cached_detector[inject.prefix_to_name[inst]].response for inst in names]

	result = dict.fromkeys((frozenset(instruments) for n in xrange(2, len(inst_horiz_mapping) + 1) for instruments in iterutils.choices(tuple(inst_horiz_mapping), n)), 0.0)

	psi = gmst = 0.0
	for i in xrange(n_samples):
		theta = math.acos(random.uniform(-1., 1.))
		phi = random.uniform(0., 2. * math.pi)
		cosi2 = random.uniform(-1., 1.)**2.

		fpfc2 = numpy.array([inject.XLALComputeDetAMResponse(resp, phi, math.pi / 2. - theta, psi, gmst) for resp in resps])**2.

		# ratio of inverse SNR to distance for each instrument
		snr_times_D = 8. * DH * numpy.dot(fpfc2, numpy.array([(1. + cosi2)**2. / 4., cosi2]))**0.5

		# the volume visible to each instrument given the
		# requirement that a source be above the SNR threshold
		# (omitting factor of 4/3 pi)
		V_at_snr_threshold = (snr_times_D / snr_threshold)**3.

		# order[0] is index of instrument that can see sources the
		# farthest, etc.
		order = sorted(range(len(V_at_snr_threshold)), key = V_at_snr_threshold.__getitem__, reverse = True)

		# instrument combination and volume of space visible to
		# that combination given the requirement that a source be
		# above the SNR threshold in that combination (omitting
		# factor or 4/3 pi).  sequence of instrument combinations
		# is left as a generator expression for lazy evaluation
		instruments = (frozenset(names[i] for i in order[:n]) for n in xrange(2, len(order) + 1))
		V = tuple(V_at_snr_threshold[i] for i in order[1:])

		# for each instrument combination, probability that a
		# source visible to at least two instruments is visible to
		# that combination
		P = [x / V[0] for x in V]

		# for each instrument combination, probability that a
		# source visible to at least two instruments is visible to
		# that combination and no other instruments
		P = [P[i] - P[i + 1] for i in xrange(len(P) - 1)] + [P[-1]]

		# accumulate result
		for key, p in zip(instruments, P):
			result[key] += p
	for key in result:
		result[key] /= n_samples

	#
	# make sure it's normalized
	#

	total = sum(sorted(result.values()))
	assert abs(total - 1.) < 1e-13
	for key in result:
		result[key] /= total

	#
	# done
	#

	return result


#
# =============================================================================
#
#                       False Alarm Book-Keeping Object
#
# =============================================================================
#


def binned_likelihood_rates_from_samples(samples, bins_per_decade = 250.0, min_bins = 1000, limits = None):
	"""
	Construct and return a BinnedArray containing a histogram of a
	sequence of samples.  If limits is None (default) then the limits
	of the binning will be determined automatically from the sequence,
	otherwise limits is expected to be a tuple or other two-element
	sequence providing the (low, high) limits, and in that case the
	sequence can be a generator.
	"""
	if limits is None:
		# add a factor of 10 of padding for the smoothing that will
		# be done later
		lo, hi = min(samples) / 10.0, max(samples) * 10.0
	else:
		lo, hi = limits
		if lo >= hi:
			raise ValueError("limits out of order (%g, %g)" % limits)
	nbins = max(int(round(bins_per_decade * math.log10(hi / lo))), min_bins)
	if nbins < 1:
		raise ValueError("bins_per_decade (%g) too small for limits (%g, %g)" % (nbins, lo, hi))
	signal_rates = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(lo, hi, nbins),)))
	noise_rates = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(lo, hi, nbins),)))
	for lamb, lnP_signal, lnP_noise in samples:
		signal_rates[lamb,] += math.exp(lnP_signal)
		noise_rates[lamb,] += math.exp(lnP_noise)
	return signal_rates, noise_rates


#
# Class to compute ranking statistic PDFs for background-like and
# signal-like populations
#
# FIXME:  this is really close to just being another subclass of
# CoincParamsDistributions.  consider the wisdom of rewriting it to be such
#


class RankingData(object):
	ligo_lw_name_suffix = u"gstlal_inspiral_rankingdata"

	#
	# Range of likelihood ratios to track in PDFs
	#

	likelihood_ratio_limits = (math.exp(-3.0), math.exp(230.0))

	#
	# e-fold scale on which to smooth likelihood ratio PDFs
	#

	likelihood_ratio_smoothing_scale = 1. / 8.

	#
	# Threshold at which FAP & FAR normalization will occur
	#

	likelihood_ratio_threshold = math.exp(2)


	def __init__(self, coinc_params_distributions, instruments = None, process_id = None, verbose = False):
		self.background_likelihood_rates = {}
		self.background_likelihood_pdfs = {}
		self.signal_likelihood_rates = {}
		self.signal_likelihood_pdfs = {}
		self.process_id = process_id

		# bailout out used by .from_xml() class method to get an
		# uninitialized instance
		if coinc_params_distributions is None:
			return

		# get default instruments from whatever we have SNR PDFs for
		if instruments is None:
			instruments = set()
			for key in coinc_params_distributions.snr_joint_pdf_cache:
				instruments |= set(instrument for instrument, distance in key)
		instruments = tuple(instruments)

		# calculate all of the possible ifo combinations with at least
		# 2 detectors in order to get the joint likelihood pdfs
		likelihoodratio_func = snglcoinc.LikelihoodRatio(coinc_params_distributions)
		threads = []
		for key in [frozenset(ifos) for n in range(2, len(instruments) + 1) for ifos in iterutils.choices(instruments, n)]:
			if verbose:
				print >>sys.stderr, "computing signal and noise likelihood PDFs for %s" % ", ".join(sorted(key))
			q = multiprocessing.queues.SimpleQueue()
			p = multiprocessing.Process(target = lambda: q.put(binned_likelihood_rates_from_samples(self.likelihoodratio_samples(coinc_params_distributions.random_params(key).next, likelihoodratio_func, coinc_params_distributions.lnP_signal, coinc_params_distributions.lnP_noise), limits = self.likelihood_ratio_limits)))
			p.start()
			threads.append((p, q, key))
		while threads:
			p, q, key = threads.pop(0)
			self.signal_likelihood_rates[key], self.background_likelihood_rates[key] = q.get()
			p.join()
			if p.exitcode:
				raise Exception("likelihood ratio sampling thread failed")
		if verbose:
			print >>sys.stderr, "done computing likelihood PDFs"

		#
		# propogate knowledge of the background event rates through
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
				binnedarray.array *= coinc_params_distributions.background_rates["instruments"][coinc_params_distributions.instrument_categories.category(instruments),] / binnedarray.array.sum()

		#
		# propogate instrument combination priors through to
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
				binnedarray.array *= coinc_params_distributions.injection_rates["instruments"][coinc_params_distributions.instrument_categories.category(instruments),] / binnedarray.array.sum()

		#
		# compute combined rates
		#

		self._compute_combined_rates()

		#
		# populate the ranking statistic PDF arrays from the counts
		#

		self.finish()

	@staticmethod
	def likelihoodratio_samples(random_params_func, likelihoodratio_func, lnP_signal_func, lnP_noise_func, nsamples = 8000000):
		for i in xrange(nsamples):
			params, lnP_params = random_params_func()
			lamb = likelihoodratio_func(params)
			assert not math.isnan(lamb)
			yield lamb, lnP_signal_func(params) - lnP_params, lnP_noise_func(params) - lnP_params

	def _compute_combined_rates(self):
		#
		# compute combined noise and signal rates
		#

		try:
			del self.background_likelihood_rates[None]
		except KeyError:
			pass
		total_rate = self.background_likelihood_rates.itervalues().next().copy()
		total_rate.array[:] = sum(binnedarray.array for binnedarray in self.background_likelihood_rates.values())
		self.background_likelihood_rates[None] = total_rate

		try:
			del self.signal_likelihood_rates[None]
		except KeyError:
			pass
		total_rate = self.signal_likelihood_rates.itervalues().next().copy()
		total_rate.array[:] = sum(binnedarray.array for binnedarray in self.signal_likelihood_rates.values())
		self.signal_likelihood_rates[None] = total_rate

	def finish(self, verbose = False):
		self.background_likelihood_pdfs.clear()
		self.signal_likelihood_pdfs.clear()
		def build_pdf(binnedarray, likelihood_ratio_threshold, smoothing_efolds):
			# copy counts into pdf array, 0 the overflow bins, and smooth
			pdf = binnedarray.copy()
			pdf.array[0] = pdf.array[-1] = 0.
			bins_per_efold = pdf.bins[0].n / math.log(pdf.bins[0].max / pdf.bins[0].min)
			kernel = rate.gaussian_window(bins_per_efold * smoothing_efolds)
			# FIXME:  this algorithm should be implemented in a
			# reusable function
			result = numpy.zeros_like(pdf.array)
			while pdf.array.any():
				workspace = numpy.copy(pdf.array)
				cutoff = abs(workspace[abs(workspace) > 0]).min() * 1e4
				pdf.array[abs(pdf.array) <= cutoff] = 0.
				workspace[abs(workspace) > cutoff] = 0.
				rate.filter_array(workspace, kernel)
				workspace[abs(workspace) < abs(workspace).max() * 1e-14] = 0.
				result += workspace
			pdf.array = result
			# zero the PDF below the threshold.  need to
			# make sure the bin @ threshold is also 0'ed
			pdf[:likelihood_ratio_threshold,] = 0.
			pdf[likelihood_ratio_threshold,] = 0.
			# convert to normalized PDF
			pdf.to_pdf()
			return pdf
		if verbose:
			progressbar = progress.ProgressBar(text = "Computing Lambda PDFs", max = len(self.background_likelihood_rates) + len(self.signal_likelihood_rates))
			progressbar.show()
		else:
			progressbar = None
		for key, binnedarray in self.background_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s noise model likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.background_likelihood_pdfs[key] = build_pdf(binnedarray, self.likelihood_ratio_threshold, self.likelihood_ratio_smoothing_scale)
			if progressbar is not None:
				progressbar.increment()
		for key, binnedarray in self.signal_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s signal model likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.signal_likelihood_pdfs[key] = build_pdf(binnedarray, self.likelihood_ratio_threshold, self.likelihood_ratio_smoothing_scale)
			if progressbar is not None:
				progressbar.increment()

	def __iadd__(self, other):
		snglcoinc.CoincParamsDistributions.addbinnedarrays(self.background_likelihood_rates, other.background_likelihood_rates, self.background_likelihood_pdfs, other.background_likelihood_pdfs)
		snglcoinc.CoincParamsDistributions.addbinnedarrays(self.signal_likelihood_rates, other.signal_likelihood_rates, self.signal_likelihood_pdfs, other.signal_likelihood_pdfs)
		return self

	@classmethod
	def from_xml(cls, xml, name):
		# find the root of the XML tree containing the
		# serialization of this object
		xml, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:%s" % (name, cls.ligo_lw_name_suffix)]

		# create a mostly uninitialized instance
		self = cls(None, {}, process_id = ligolw_param.get_pyvalue(xml, u"process_id"))

		# pull out the likelihood count and PDF arrays
		def reconstruct(xml, prefix, target_dict):
			for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and ("_%s" % prefix) in elem.getAttribute(u"Name")]:
				ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.getAttribute(u"Name").split("_")[0]))
				target_dict[ifo_set] = rate.binned_array_from_xml(ba_elem, ba_elem.getAttribute(u"Name").split(":")[0])
		reconstruct(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		reconstruct(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		reconstruct(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		reconstruct(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)

		assert set(self.background_likelihood_rates) == set(self.background_likelihood_pdfs)
		assert set(self.signal_likelihood_rates) == set(self.signal_likelihood_pdfs)
		assert set(self.background_likelihood_rates) == set(self.signal_likelihood_rates)

		self._compute_combined_rates()

		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(ligolw_param.new_param(u"process_id", u"ilwd:char", self.process_id))
		def store(xml, prefix, source_dict):
			for key, binnedarray in source_dict.items():
				if key is not None:
					ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
					xml.appendChild(rate.binned_array_to_xml(binnedarray, u"%s_%s" % (ifostr, prefix)))
		store(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		store(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		store(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		store(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)

		return xml


#
# Class to compute false-alarm probabilities and false-alarm rates from
# ranking statistic PDFs
#


class FAPFAR(object):
	def __init__(self, ranking_stat_pdfs, count_above_threshold, threshold, livetime = None):
		# None is OK, but then can only compute FAPs, not FARs
		self.livetime = livetime
		# dictionary keyed by instrument combination
		assert set(count_above_threshold) >= set(ranking_stat_pdfs), "incomplete count_above_threshold:  missing counts for %s" % (set(ranking_stat_pdfs) - set(count_above_threshold))
		assert set(ranking_stat_pdfs) >= set(key for key, value in count_above_threshold.items() if value != 0), "incomplete ranking_stat_pdfs:  have count-above-thresholds for %s" % (set(key for key, value in count_above_threshold.items() if value != 0) - set(ranking_stat_pdfs))
		# make copy as normal dictionary
		self.count_above_threshold = dict(count_above_threshold)

		self.cdf_interpolator = {}
		self.ccdf_interpolator = {}

		_ranks = None
		for instruments, binnedarray in ranking_stat_pdfs.items():
			instruments_name = ", ".join(sorted(instruments)) if instruments is not None else "combined"
			assert not numpy.isnan(binnedarray.array).any(), "%s likelihood ratio PDF contains NaNs" % instruments_name
			assert not (binnedarray.array < 0.0).any(), "%s likelihood ratio PDF contains negative values" % instruments_name

			# convert PDF into probability mass per bin, being
			# careful with bins that are infinite in size
			ranks, = binnedarray.bins.upper()
			drank = binnedarray.bins.volumes()
			weights = (binnedarray.array * drank).compress(numpy.isfinite(drank))
			ranks = ranks.compress(numpy.isfinite(drank))
			if _ranks is None:
				_ranks = ranks
			elif (_ranks != ranks).any():
				raise ValueError("incompatible binnings in ranking statistics PDFs")
			# cumulative distribution function and its
			# complement.  it's numerically better to recompute
			# the ccdf by reversing the array of weights than
			# trying to subtract the cdf from 1.
			cdf = weights.cumsum()
			cdf /= cdf[-1]
			ccdf = weights[::-1].cumsum()[::-1]
			ccdf /= ccdf[0]
			assert not numpy.isnan(cdf).any(), "%s likelihood ratio CDF contains NaNs" % instruments_name
			assert not numpy.isnan(ccdf).any(), "%s likelihood ratio CCDF contains NaNs" % instruments_name
			assert ((0. <= cdf) & (cdf <= 1.)).all(), "%s likelihood ratio CDF failed to be normalized" % instruments_name
			assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "%s likelihood ratio CCDF failed to be normalized" % instruments_name
			# cdf boundary condition:  cdf = 1/e at the ranking
			# statistic threshold so that
			# self.far_from_rank(threshold) * livetime =
			# observed count of events above threshold.
			ccdf *= 1. - 1. / math.e
			cdf *= 1. - 1. / math.e
			cdf += 1. / math.e
			# make cdf + ccdf == 1
			#s = cdf[:-1] + ccdf[1:]
			#cdf[:-1] /= s
			#ccdf[1:] /= s
			# one last check that normalization is OK
			assert (abs(1. - (cdf[:-1] + ccdf[1:])) < 1e-12).all(), "%s likelihood ratio CDF + CCDF != 1 (max error = %g)" % (instruments_name, abs(1. - (cdf[:-1] + ccdf[1:])).max())
			# build interpolators
			self.cdf_interpolator[instruments] = interpolate.interp1d(ranks, cdf)
			self.ccdf_interpolator[instruments] = interpolate.interp1d(ranks, ccdf)
			# make sure boundary condition survives interpolator
			assert abs(float(self.cdf_interpolator[instruments](threshold)) - 1. / math.e) < 1e-14, "%s CDF interpolator fails at threshold (= %g)" % (instruments_name, float(self.cdf_interpolator[instruments](threshold)))
		if _ranks is None:
			raise ValueError("no ranking statistic PDFs")

		# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
		self.minrank = max(threshold, min(_ranks))
		self.maxrank = max(_ranks)

	def fap_from_rank(self, rank):
		rank = max(self.minrank, min(self.maxrank, rank))
		fap = float(self.ccdf_interpolator[None](rank))
		return fap_after_trials(fap, self.count_above_threshold[None])

	def far_from_rank(self, rank):
		assert self.livetime is not None, "cannot compute FAR without livetime"
		rank = max(self.minrank, min(self.maxrank, rank))
		# true-dismissal probability = 1 - false-alarm probability
		tdp = float(self.cdf_interpolator[None](rank))
		try:
			log_tdp = math.log(tdp)
		except ValueError:
			# TDP = 0 --> FAR = +inf
			return PosInf
		if log_tdp >= -1e-9:
			# rare event:  avoid underflow by using log1p(-FAP)
			log_tdp = math.log1p(-float(self.ccdf_interpolator[None](rank)))
		return self.count_above_threshold[None] * -log_tdp / self.livetime

	def assign_faps(self, connection):
		# assign false-alarm probabilities
		# FIXME:  choose a function name more likely to be unique?
		# FIXME:  abusing false_alarm_rate column, move for a
		# false_alarm_probability column??
		connection.create_function("fap", 1, self.fap_from_rank)
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
	)
""")

	def assign_fars(self, connection):
		# assign false-alarm rates
		# FIXME:  choose a function name more likely to be unique?
		connection.create_function("far", 1, self.far_from_rank)
		connection.cursor().execute("""
UPDATE
	coinc_inspiral
SET
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


#
# =============================================================================
#
#                                    Other
#
# =============================================================================
#


def get_live_time(seglists, verbose = False):
	livetime = float(abs(vote((segs for instrument, segs in seglists.items() if instrument != "H2"), 2)))
	if verbose:
		print >> sys.stderr, "Livetime: %.3g s" % livetime
	return livetime


def get_live_time_segs_from_search_summary_table(connection, program_name = "gstlal_inspiral"):
	xmldoc = dbtables.get_xml(connection)
	farsegs = ligolw_search_summary.segmentlistdict_fromsearchsummary(xmldoc, program_name).coalesce()
	xmldoc.unlink()
	return farsegs


#
# =============================================================================
#
#                            Event Rate Posteriors
#
# =============================================================================
#


def RatesLnPDF((Rf, Rb), f_over_b, lnpriorfunc = lambda Rf, Rb: -0.5 * math.log(Rf * Rb)):
	"""
	Compute the log probability density of the foreground and
	background rates given by equation (21) in Farr et al., "Counting
	and Confusion:  Bayesian Rate Estimation With Multiple
	Populations", arXiv:1302.5341.  The default prior is that specified
	in the paper but it can be overridden with the lnpriorfunc keyword
	argument (giving a function that returns the natural log of the
	prior given Rf, Rb).
	"""
	if Rf <= 0. or Rb <= 0.:
		return NegInf
	return numpy.log1p((Rf / Rb) * f_over_b).sum() + len(f_over_b) * math.log(Rb) - (Rf + Rb) + lnpriorfunc(Rf, Rb)


def maximum_likelihood_rates(f_over_b):
	def F(x):
		return -RatesLnPDF(x, f_over_b)
	from scipy.optimize import fmin
	return fmin(F, (1.0, float(len(f_over_b))), disp = True)


def run_mcmc(n_walkers, n_dim, n_samples_per_walker, lnprobfunc, pos0 = None, args = (), n_burn = 100, progressbar = None):
	"""
	A generator function that yields samples distributed according to a
	user-supplied probability density function that need not be
	normalized.  lnprobfunc computes and returns the natural logarithm
	of the probability density, up to a constant offset.  n_dim sets
	the number of dimensions of the parameter space over which the PDF
	is defined and args gives any additional arguments to be passed to
	lnprobfunc, whose signature must be

		ln(P) = lnprobfunc(X, *args)

	where X is a numpy array of length n_dim.

	The generator yields a total of n_walkers * n_samples_per_walker
	samples drawn from the n_dim-dimensional parameter space.  Each
	sample is returned as a numpy array.

	mean and stdev adjust the Gaussian random number generator used to
	set the initial co-ordinates of the walkers.  n_burn iterations of
	the MCMC sampler will be executed and discarded to allow the system
	to stabilize before samples are yielded to the calling code.
	"""
	#
	# construct a sampler
	#

	sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprobfunc, args = args, threads = 2)

	#
	# set walkers at initial positions
	#

	# FIXME:  implement
	assert pos0 is not None, "auto-selection of initial positions not implemented"

	#
	# burn-in:  run for a while to get better initial positions
	#

	pos0, ignored, ignored = sampler.run_mcmc(pos0, n_burn, storechain = False)
	if progressbar is not None:
		progressbar.increment(delta = n_burn)
	if sampler.acceptance_fraction.min() < 0.4:
		print >>sys.stderr, "\nwarning:  low burn-in acceptance fraction (min = %g)" % sampler.acceptance_fraction.min()

	#
	# reset and yield positions distributed according to the supplied
	# PDF
	#

	sampler.reset()
	for coordslist, ignored, ignored in sampler.sample(pos0, iterations = n_samples_per_walker, storechain = False):
		for coords in coordslist:
			yield coords
		if progressbar is not None:
			progressbar.increment()
	if sampler.acceptance_fraction.min() < 0.5:
		print >>sys.stderr, "\nwarning:  low sampler acceptance fraction (min %g)" % sampler.acceptance_fraction.min()


def binned_rates_from_samples(samples):
	"""
	Construct and return a BinnedArray containing a histogram of a
	sequence of samples.  If limits is None (default) then the limits
	of the binning will be determined automatically from the sequence,
	otherwise limits is expected to be a tuple or other two-element
	sequence providing the (low, high) limits, and in that case the
	sequence can be a generator.
	"""
	lo, hi = math.floor(samples.min()), math.ceil(samples.max())
	nbins = len(samples) // 600
	binnedarray = rate.BinnedArray(rate.NDBins((rate.LinearBins(lo, hi, nbins),)))
	for sample in samples:
		binnedarray[sample,] += 1.
	rate.filter_array(binnedarray.array, rate.gaussian_window(3))
	numpy.clip(binnedarray.array, 0.0, PosInf, binnedarray.array)
	return binnedarray


def calculate_rate_posteriors(ranking_data, likelihood_ratios, progressbar = None):
	"""
	FIXME:  document this
	"""
	#
	# check for funny input
	#

	if any(math.isnan(lr) for lr in likelihood_ratios):
		raise ValueError("NaN likelihood ratio encountered")
	# FIXME;  clip likelihood ratios to some maximum value?
	if any(math.isinf(lr) for lr in likelihood_ratios):
		raise ValueError("infinite likelihood ratio encountered")

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.  since order is irrelevant in what follows, construct the
	# array in ascending order for better numerical behaviour in the
	# cost function.  the sort order is stored in a look-up table in
	# case we want to do something with it later (the Farr et al.
	# paper provides a technique for assessing the probability that
	# each event individually is a signal and if we ever implement that
	# here then we need to have not lost the original event order).
	#

	order = range(len(likelihood_ratios))
	order.sort(key = likelihood_ratios.__getitem__)
	f_over_b = numpy.array([ranking_data.signal_likelihood_pdfs[None][likelihood_ratios[index],] / ranking_data.background_likelihood_pdfs[None][likelihood_ratios[index],] for index in order])

	# remove NaNs.  these occur because the ranking statistic PDFs have
	# been zeroed at the cut-off and some events get pulled out of the
	# database with ranking statistic values in that bin
	#
	# FIXME:  re-investigate the decision to zero the bin at threshold.
	# the original motivation for doing it might not be there any
	# longer
	f_over_b = f_over_b[~numpy.isnan(f_over_b)]
	# safety check
	if numpy.isinf(f_over_b).any():
		raise ValueError("infinity encountered in ranking statistic PDF ratios")

	#
	# run MCMC sampler to generate (foreground rate, background rate)
	# samples.
	#

	ndim = 2
	nwalkers = 10 * 2 * ndim	# must be even and >= 2 * ndim
	nsample = 40000
	nburn = 1000

	if progressbar is not None:
		progressbar.max = nsample + nburn
		progressbar.show()

	if True:
		pos0 = numpy.zeros((nwalkers, ndim), dtype = "double")
		pos0[:,0] = numpy.random.exponential(scale = 1., size = (nwalkers,))
		pos0[:,1] = numpy.random.poisson(lam = len(likelihood_ratios), size = (nwalkers,))
		samples = numpy.empty((nwalkers * nsample, ndim), dtype = "double")
		for i, sample in enumerate(run_mcmc(nwalkers, ndim, nsample, RatesLnPDF, n_burn = nburn, args = (f_over_b,), pos0 = pos0, progressbar = progressbar)):
			samples[i] = sample
		import pickle
		pickle.dump(samples, open("rate_posterior_samples.pickle", "w"))
	else:
		import pickle
		samples = pickle.load(open("rate_posterior_samples.pickle"))
		progressbar.increment(delta = progressbar.max)
	if samples.min() < 0:
		raise ValueError("MCMC sampler yielded negative rate(s)")

	#
	# compute marginalized PDFs for the foreground and background rates
	#

	Rf_pdf = binned_rates_from_samples(samples[:,0])
	Rf_pdf.to_pdf()
	Rb_pdf = binned_rates_from_samples(samples[:,1])
	Rb_pdf.to_pdf()

	#
	# done
	#

	return Rf_pdf, Rb_pdf


def confidence_interval_from_binnedarray(binned_array, confidence = 0.95):
	"""
	Constructs a confidence interval based on a BinnedArray object
	containing a normalized 1-D PDF.  Returns the tuple (mode, lower bound,
	upper bound).
	"""
	# check for funny input
	if numpy.isnan(binned_array.array).any():
		raise ValueError("NaNs encountered in rate PDF")
	if numpy.isinf(binned_array.array).any():
		raise ValueError("infinities encountered in rate PDF")
	if (binned_array.array < 0.).any():
		raise ValueError("negative values encountered in rate PDF")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be in [0, 1]")

	mode_index = numpy.argmax(binned_array.array)

	centres, = binned_array.centres()
	upper = binned_array.bins[0].upper()
	lower = binned_array.bins[0].lower()
	bin_widths = upper - lower
	if (bin_widths <= 0.).any():
		raise ValueError("rate PDF bin sizes must be positive")
	assert not numpy.isinf(bin_widths).any(), "infinite bin sizes not supported"
	P = binned_array.array * bin_widths
	if abs(P.sum() - 1.0) > 1e-13:
		raise ValueError("rate PDF is not normalized")

	li = ri = mode_index
	P_sum = P[mode_index]
	while P_sum < confidence:
		if li <= 0 and ri >= len(P) - 1:
			raise ValueError("failed to achieve requested confidence")
		P_li = P[li - 1] if li > 0 else 0.
		P_ri = P[ri + 1] if ri < len(P) - 1 else 0.
		assert P_li >= 0. and P_ri >= 0.
		if P_li > P_ri:
			P_sum += P_li
			li -= 1
		elif P_ri > P_li:
			P_sum += P_ri
			ri += 1
		else:
			P_sum += P_li + P_ri
			li = max(li - 1, 0)
			ri = min(ri + 1, len(P) - 1)
	return centres[mode_index], lower[li], upper[ri]
