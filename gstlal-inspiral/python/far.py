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
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments
from glue import segments
from glue.segmentsUtils import vote
from gstlal import emcee
from pylal import inject
from pylal import progress
from pylal import rate
from pylal import snglcoinc


class DefaultContentHandler(ligolw.LIGOLWContentHandler):
	pass
ligolw_array.use_in(DefaultContentHandler)
ligolw_param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)


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
#                             Trials Table Object
#
# =============================================================================
#


#
# Trials table
#


class Trials(object):
	def __init__(self, count = 0, count_below_thresh = 0, thresh = None):
		self.count = count
		self.count_below_thresh = count_below_thresh
		self.thresh = thresh

	def __add__(self, other):
		out = type(self)(self.count, self.count_below_thresh, self.thresh)
		out.count +=  other.count
		out.count_below_thresh += other.count_below_thresh
		assert(out.thresh == other.thresh)
		return out


class TrialsTable(dict):
	"""
	A class to store the trials table from a coincident inspiral search
	with the intention of computing the false alarm probabiliy of an event after N
	trials.  This is a subclass of dict.  The trials table is keyed by the
	detectors that partcipated in the coincidence.
	"""
	class TrialsTableTable(lsctables.table.Table):
		tableName = "gstlal_trials:table"
		validcolumns = {
			"ifos": "lstring",
			"count": "int_8s",
			"count_below_thresh": "int_8s",
			"thresh": "real_8"
		}
		class RowType(object):
			__slots__ = ("ifos", "count", "count_below_thresh", "thresh")

			def get_ifos(self):
				return lsctables.instrument_set_from_ifos(self.ifos)

			def set_ifos(self, ifos):
				self.ifos = lsctables.ifos_from_instrument_set(ifos)

			@property
			def key(self):
				return frozenset(self.get_ifos())

			@classmethod
			def from_item(cls, (ifos, trials)):
				self = cls()
				self.set_ifos(ifos)
				self.count = trials.count
				self.count_below_thresh = trials.count_below_thresh
				self.thresh = trials.thresh
				return self

	def initialize_from_sngl_ifos(self, ifos, count = 0, count_below_thresh = 0, thresh = None):
		"""
		for all possible combinations of 2 or more from ifos initialize ourself to provided values
		"""
		for n in range(2, len(ifos) +	1):
			for ifo in iterutils.choices(ifos, n):
				self[frozenset(ifo)] = Trials(count, count_below_thresh, thresh)

	def get_sngl_ifos(self):
		out = set()
		for ifos in self:
			for ifo in ifos:
				out.add(ifo)
		return tuple(out)

	def __add__(self, other):
		out = type(self)()
		for k in self:
			out[k] = type(self[k])(self[k].count, self[k].count_below_thresh, self[k].thresh)
		for k in other:
			try:
				out[k] += other[k]
			except KeyError:
				out[k] = type(other[k])(other[k].count, other[k].count_below_thresh, other[k].thresh)
		return out

	def increment_count(self, n):
		"""
		Increment all keys by n
		"""
		for k in self:
			self[k].count += n

	def num_nonzero_count(self):
		return len([k for k in self if self[k].count != 0])

	def set_thresh(self, thresh = None):
		for k in self:
			self[k].thresh = thresh

	@classmethod
	def from_xml(cls, xml):
		"""
		A class method to create a new instance of a TrialsTable from
		an xml representation of it.
		"""
		self = cls()
		for row in lsctables.table.get_table(xml, self.TrialsTableTable.tableName):
			self[row.key] = Trials(row.count, row.count_below_thresh, row.thresh)
		return self

	def to_xml(self):
		"""
		A method to write this instance of a trials table to an xml
		representation.
		"""
		xml = lsctables.New(self.TrialsTableTable)
		for item in self.items():
			xml.append(xml.RowType.from_item(item))
		return xml


lsctables.TableByName[lsctables.table.StripTableName(TrialsTable.TrialsTableTable.tableName)] = TrialsTable.TrialsTableTable


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
		params = dict(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events)
		# don't allow both H1 and H2 to participate in the same
		# coinc.  if both have participated favour H1
		if "H2_snr_chi" in params and "H1_snr_chi" in params:
			del params["H2_snr_chi"]
		# FIXME:  currently this is not used to form part of the
		# ranking statistic's parameters.  it *is* used to track
		# non-coincident singles rates and zero-lag coinc rates
		params["instruments"] = (ThincaCoincParamsDistributions.instrument_categories.category(event.ifo for event in events),)
		return params

	def P_noise(self, params):
		if params is None:
			return None
		P = 1.0
		for name, value in params.items():
			if name.endswith("_snr_chi"):
				P *= self.background_pdf_interp[name](*value)
		return P

	def P_signal(self, params):
		if params is None:
			return None

		# (instrument, snr) pairs sorted alphabetically by instrument name
		snrs = sorted((name.split("_")[0], value[0]) for name, value in params.items() if name.endswith("_snr_chi"))
		# retrieve the SNR PDF
		snr_pdf = self.get_snr_joint_pdf(dict((instrument, self.horizon_distances[instrument]) for instrument, rho in snrs))
		# evaluate it (snrs are alphabetical by instrument)
		P = snr_pdf(*tuple(rho for instrument, rho in snrs))

		for name, value in params.items():
			if name.endswith("_snr_chi"):
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
			if verbose:
				progressbar = progress.ProgressBar(instrument)
			else:
				progressbar = None
			binarr = self.background_rates["%s_snr_chi" % instrument]

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
			for i, snr in enumerate(snrs):
				p = math.exp(-snr**2 / 2. + snrs[0]**2 / 2. + math.log(n))
				p += (transition / snr)**6 * math.exp(-transition**2 / 2. + snrs[0]**2 / 2. + math.log(n)) # Softer fall off above some transition SNR for numerical reasons
				for chi2_over_snr2 in chi2_over_snr2s:
					new_binarr[snr, chi2_over_snr2] += p
				if progressbar is not None:
					progressbar.update((i + 1.0) / len(snrs))
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
			if verbose:
				progressbar = progress.ProgressBar(instrument)
			else:
				progressbar = None
			binarr = target_dict["%s_snr_chi" % instrument]

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
			for i, snr in enumerate(snrs):
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
					progressbar.update((i + 1.0) / len(snrs))
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
		progressbar = progress.ProgressBar("%s SNR joint PDF" % ", ".join(names))
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
		snr_start = snr_times_D[axis] / (snr_times_D.min() / snr_min)

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
			progressbar.update((i + 1.) / n_samples)

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
# Class to handle the computation of FAPs/FARs
#


class LocalRankingData(object):
	def __init__(self, livetime_seg, coinc_params_distributions, trials_table = None):
		self.distributions = coinc_params_distributions
		self.likelihoodratio = snglcoinc.LikelihoodRatio(coinc_params_distributions)
		if trials_table is None:
			self.trials_table = TrialsTable()
		else:
			self.trials_table = trials_table
		self.background_likelihood_rates = {}
		self.background_likelihood_pdfs = {}
		self.signal_likelihood_rates = {}
		self.signal_likelihood_pdfs = {}
		self.livetime_seg = livetime_seg

	def __iadd__(self, other):
		self.distributions += other.distributions
		self.trials_table += other.trials_table
		if self.livetime_seg[0] is None and other.livetime_seg[0] is not None:
			minstart = other.livetime_seg[0]
		elif self.livetime_seg[0] is not None and other.livetime_seg[0] is None:
			minstart = self.livetime_seg[0]
		# correctly handles case where both or neither are None
		else:
			minstart = min(self.livetime_seg[0], other.livetime_seg[0])
		# None is always less than everything else, so this is okay
		maxend = max(self.livetime_seg[1], other.livetime_seg[1])
		self.livetime_seg = segments.segment(minstart, maxend)

		self.distributions.addbinnedarrays(self.background_likelihood_rates, other.background_likelihood_rates, self.background_likelihood_pdfs, other.background_likelihood_pdfs)
		self.distributions.addbinnedarrays(self.signal_likelihood_rates, other.signal_likelihood_rates, self.signal_likelihood_pdfs, other.signal_likelihood_pdfs)

		return self

	@classmethod
	def from_xml(cls, xml, name = u"gstlal_inspiral_likelihood"):
		llw_elem, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_FAR" % name]
		distributions = ThincaCoincParamsDistributions.from_xml(llw_elem, name)
		# the code that writes these things has put the
		# livetime_seg into the out segment in the search_summary
		# table.  uninitialized segments got recorded as
		# [None,None)
		# FIXME:  move livetime info into segment tables, and
		# instead of doing what follows attach a segment list name
		# to this class that gets recorded as a Param and don't try
		# to store livetime info in this class at all
		try:
			search_summary_table = lsctables.table.get_table(xml, lsctables.SearchSummaryTable.tableName)
		except ValueError:
			livetime_seg = segments.segment(None, None)
		else:
			livetime_seg, = (row.get_out() for row in search_summary_table if row.process_id == distributions.process_id)
		self = cls(livetime_seg, coinc_params_distributions = distributions, trials_table = TrialsTable.from_xml(llw_elem))

		# pull out the joint likelihood arrays if they are present
		def reconstruct(xml, prefix, target_dict):
			for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and ("_%s" % prefix) in elem.getAttribute(u"Name")]:
				ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.getAttribute(u"Name").split("_")[0]))
				target_dict[ifo_set] = rate.binned_array_from_xml(ba_elem, ba_elem.getAttribute(u"Name").split(":")[0])
		reconstruct(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		reconstruct(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		reconstruct(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		reconstruct(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)

		return self

	@classmethod
	def from_filenames(cls, filenames, name = u"gstlal_inspiral_likelihood", contenthandler = DefaultContentHandler, verbose = False):
		self = LocalRankingData.from_xml(ligolw_utils.load_filename(filenames[0], contenthandler = contenthandler, verbose = verbose), name = name)
		for f in filenames[1:]:
			self += LocalRankingData.from_xml(ligolw_utils.load_filename(f, contenthandler = contenthandler, verbose = verbose), name = name)
		return self

	def to_xml(self, name = u"gstlal_inspiral_likelihood"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_FAR" % name})
		xml.appendChild(self.trials_table.to_xml())
		xml.appendChild(self.distributions.to_xml(name))
		def store(xml, prefix, source_dict):
			for key, binnedarray in source_dict.items():
				ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
				xml.appendChild(rate.binned_array_to_xml(binnedarray, u"%s_%s" % (ifostr, prefix)))
		store(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		store(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		store(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		store(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)

		return xml

	def finish(self):
		self.background_likelihood_pdfs.clear()
		self.signal_likelihood_pdfs.clear()
		for key, binnedarray in self.background_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s noise model likelihood ratio counts contain NaNs" % key
			# copy counts
			self.background_likelihood_pdfs[key] = binnedarray.copy()
			# smooth on a scale ~= 1/4 unit of SNR
			bins_per_efold = binnedarray.bins[0].n / math.log(binnedarray.bins[0].max / binnedarray.bins[0].min)
			rate.filter_array(self.background_likelihood_pdfs[key].array, rate.gaussian_window(.25 * bins_per_efold))
			# guard against round-off in FFT convolution
			# yielding negative probability densities
			numpy.clip(self.background_likelihood_pdfs[key].array, 0.0, PosInf, self.background_likelihood_pdfs[key].array)
			# convert to normalized PDF
			self.background_likelihood_pdfs[key].to_pdf()
		for key, binnedarray in self.signal_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s signal model likelihood ratio counts contain NaNs" % key
			# copy counts
			self.signal_likelihood_pdfs[key] = binnedarray.copy()
			# smooth on a scale ~= 1/4 unit of SNR
			bins_per_efold = binnedarray.bins[0].n / math.log(binnedarray.bins[0].max / binnedarray.bins[0].min)
			rate.filter_array(self.signal_likelihood_pdfs[key].array, rate.gaussian_window(.25 * bins_per_efold))
			# guard against round-off in FFT convolution
			# yielding negative probability densities
			numpy.clip(self.signal_likelihood_pdfs[key].array, 0.0, PosInf, self.signal_likelihood_pdfs[key].array)
			# convert to normalized PDF
			self.signal_likelihood_pdfs[key].to_pdf()

	def smooth_distribution_stats(self, verbose = False):
		if verbose:
			print >>sys.stderr, "smoothing parameter distributions ...",
		self.distributions.finish()
		if verbose:
			print >>sys.stderr, "done"

	def likelihoodratio_samples(self, ln_prob, keys, ndim, nwalkers = None, nburn = 5000, nsamples = 100000):
		keys = tuple(sorted(keys))
		# FIXME:  don't know how to tune nwalkers.  must be even, and >
		# 2 * ndim
		if nwalkers is None:
			nwalkers = 24 * ndim
		for coords in run_mcmc(nwalkers, ndim, nsamples, ln_prob, n_burn = nburn):
			# coords[0::2] = rho
			# coords[1::2] = chi^2/rho^2
			lamb = self.likelihoodratio(dict(zip(keys, zip(coords[0::2], coords[1::2]))))
			if not math.isinf(lamb) and not math.isnan(lamb):	# FIXME:  is this needed?
				yield lamb

	def compute_likelihood_pdfs(self, remap, instruments = None, verbose = False):
		self.background_likelihood_rates.clear()
		self.signal_likelihood_rates.clear()

		# get default instruments from whatever we have SNR PDFs for
		if instruments is None:
			instruments = set()
			for key in self.distributions.snr_joint_pdf_cache:
				instruments |= set(instrument for instrument, distance in key)
		instruments = tuple(instruments)

		# calculate all of the possible ifo combinations with at least
		# 2 detectors in order to get the joint likelihood pdfs
		for n in range(2, len(instruments) + 1):
			for ifos in iterutils.choices(instruments, n):
				ifo_set = frozenset(ifos)
				remap_set = remap.setdefault(ifo_set, ifo_set)

				if verbose:
					print >>sys.stderr, "computing likelihood PDFs for %s remapped to %s" % (lsctables.ifos_from_instrument_set(ifo_set), lsctables.ifos_from_instrument_set(remap_set))

				# only recompute if necessary, some choices
				# might not be if a certain remap set is
				# provided
				if remap_set not in self.background_likelihood_rates:
					ndim = 2 * len(ifo_set)
					keys = frozenset("%s_snr_chi" % inst for inst in ifo_set)

					ln_prob = self.distributions.create_emcee_lnprob_wrapper(self.distributions.lnP_signal, keys)
					self.signal_likelihood_rates[remap_set] = binned_rates_from_samples(self.likelihoodratio_samples(ln_prob, keys, ndim), limits = (1e-2, 1e+100))

					ln_prob = self.distributions.create_emcee_lnprob_wrapper(self.distributions.lnP_noise, keys)
					self.background_likelihood_rates[remap_set] = binned_rates_from_samples(self.likelihoodratio_samples(ln_prob, keys, ndim), limits = (1e-2, 1e+100))

				self.background_likelihood_rates[ifo_set] = self.background_likelihood_rates[remap_set]
				self.signal_likelihood_rates[ifo_set] = self.signal_likelihood_rates[remap_set]

		self.finish()


class RankingData(object):
	def __init__(self, local_ranking_data):
		# ensure the trials tables' keys match the likelihood
		# histograms' keys
		assert set(local_ranking_data.background_likelihood_rates) == set(local_ranking_data.trials_table)

		# copy likelihood ratio counts
		# FIXME:  the raw bin counts haven't been smoothed.
		# figure out how.
		self.likelihood_rates = dict((key, value.copy()) for key, value in local_ranking_data.background_likelihood_rates.items())

		# copy trials table counts
		self.trials_table = TrialsTable()
		self.trials_table += local_ranking_data.trials_table
		self.scale = dict([(k, 1.) for k in self.trials_table])
		
		# copy livetime segment
		self.livetime_seg = local_ranking_data.livetime_seg

		self.cdf_interpolator = {}
		self.ccdf_interpolator = {}
		self.minrank = {}
		self.maxrank = {}


	@classmethod
	def from_xml(cls, xml, name = u"gstlal_inspiral"):
		llw_elem, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_ranking_data" % name]

		class fake_local_ranking_data(object):
			pass
		fake_local_ranking_data = fake_local_ranking_data()
		fake_local_ranking_data.trials_table = TrialsTable.from_xml(llw_elem)
		fake_local_ranking_data.background_likelihood_rates = {}
		for key in fake_local_ranking_data.trials_table:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			fake_local_ranking_data.background_likelihood_rates[key] = rate.binned_array_from_xml(llw_elem, ifostr)

		# the code that writes these things has put the
		# livetime_seg into the out segment in the search_summary
		# table.  uninitialized segments got recorded as
		# [None,None).
		process_id = ligolw_param.get_pyvalue(llw_elem, u"process_id")
		fake_local_ranking_data.livetime_seg, = (row.get_out() for row in lsctables.table.get_table(xml, lsctables.SearchSummaryTable.tableName) if row.process_id == process_id)

		self = cls(fake_local_ranking_data)

		return self, process_id


	def to_xml(self, process, search_summary, name = u"gstlal_inspiral"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_ranking_data" % name})
		xml.appendChild(ligolw_param.new_param(u"process_id", u"ilwd:char", process.process_id))
		xml.appendChild(self.trials_table.to_xml())
		for key, binnedarray in self.likelihood_rates.items():
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			xml.appendChild(rate.binned_array_to_xml(binnedarray, ifostr))
		assert search_summary.process_id == process.process_id
		search_summary.set_out(self.livetime_seg)
		return xml


	def __iadd__(self, other):
		our_trials = self.trials_table
		other_trials = other.trials_table

		our_keys = set(self.likelihood_rates)
		other_keys  = set(other.likelihood_rates)

		# rates that only we have are unmodified
		pass

		# rates that only the new data has get copied verbatim
		for k in other_keys - our_keys:
			self.likelihood_rates[k] = other.likelihood_rates[k].copy()

		# rates that we have and are in the new data get replaced
		# with the weighted sum, re-binned
		for k in our_keys & other_keys:
			minself, maxself, nself = self.likelihood_rates[k].bins[0].min, self.likelihood_rates[k].bins[0].max, self.likelihood_rates[k].bins[0].n
			minother, maxother, nother = other.likelihood_rates[k].bins[0].min, other.likelihood_rates[k].bins[0].max, other.likelihood_rates[k].bins[0].n
			new_likelihood_rates =  rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(min(minself, minother), max(maxself, maxother), max(nself, nother)),)))

			for x in self.likelihood_rates[k].centres()[0]:
				new_likelihood_rates[x,] += self.likelihood_rates[k][x,] * float(our_trials[k].count or 1) / ((our_trials[k].count + other_trials[k].count) or 1)
			for x in other.likelihood_rates[k].centres()[0]:
				new_likelihood_rates[x,] += other.likelihood_rates[k][x,] * float(other_trials[k].count or 1) / ((our_trials[k].count + other_trials[k].count) or 1)

			self.likelihood_rates[k] = new_likelihood_rates

		# combined trials counts
		self.trials_table += other.trials_table

		# merge livetime segments.  let the code crash if they're disjoint
		self.livetime_seg |= other.livetime_seg

		return self

	def compute_joint_cdfs(self):
		self.minrank.clear()
		self.maxrank.clear()
		self.cdf_interpolator.clear()
		self.ccdf_interpolator.clear()
		for instruments, binnedarray in self.likelihood_rates.items():
			ranks, = binnedarray.bins.lower()
			weights = binnedarray.array
			# cumulative distribution function and its
			# complement.  it's numerically better to recompute
			# the ccdf by reversing the array of weights than
			# trying to subtract the cdf from 1.
			cdf = weights.cumsum()
			cdf /= cdf[-1]
			ccdf = weights[::-1].cumsum()[::-1]
			ccdf /= ccdf[0]
			# try making ccdf + cdf == 1.  nothing really cares
			# if this identity doesn't exactly hold, but we
			# might as well avoid weirdness where we can.
			s = ccdf + cdf
			cdf /= s
			ccdf /= s
			# build interpolators
			self.cdf_interpolator[instruments] = interpolate.interp1d(ranks, cdf)
			self.ccdf_interpolator[instruments] = interpolate.interp1d(ranks, ccdf)
			# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
			self.minrank[instruments] = min(ranks)
			self.maxrank[instruments] = max(ranks)

	def fap_from_rank(self, rank, ifos):
		ifos = frozenset(ifos)
		rank = max(self.minrank[ifos], min(self.maxrank[ifos], rank))
		fap = float(self.ccdf_interpolator[ifos](rank))
		try:
			trials = max(int(self.trials_table[ifos].count), 1)
		except KeyError:
			trials = 1
		# multiply by a scale factor if available, assume scale is
		# 1 if not available.
		if ifos in self.scale:
			trials *= self.scale[ifos]
		return fap_after_trials(fap, trials)

	def far_from_rank(self, rank, ifos, scale = False):
		ifos = frozenset(ifos)
		rank = max(self.minrank[ifos], min(self.maxrank[ifos], rank))
		# true-dismissal probability = 1 - false-alarm probability
		tdp = float(self.cdf_interpolator[ifos](rank))
		if tdp == 0.:
			return PosInf
		try:
			trials = max(int(self.trials_table[ifos].count), 1)
		except KeyError:
			trials = 1
		# multiply by a scale factor if provided
		# (assume scale is 1 if disabled or not available)
		if scale and ifos in self.scale:
			trials *= self.scale[ifos]
		try:
			livetime = float(abs(self.livetime_seg))
		except TypeError:
			# we don't have a livetime segment yet.  this can
			# happen during the early start-up of an online
			# low-latency analysis
			return NaN
		return trials * -math.log(tdp) / livetime


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


#
# Function to compute the fap in a given file
#


def set_fap(Far, f, tmp_path = None, verbose = False):
	"""
	Function to set the false alarm probability for a single database
	containing the usual inspiral tables.

	Far = LocalRankingData class instance
	f = filename of the databse (e.g.something.sqlite)
	tmp_path = the local disk path to copy the database to in
		order to avoid sqlite commands over nfs
	verbose = be verbose
	"""
	# FIXME this code should be moved into a method of the LocalRankingData class once other cleaning is done
	# set up working file names
	working_filename = dbtables.get_connection_filename(f, tmp_path = tmp_path, verbose = verbose)
	connection = sqlite3.connect(working_filename)

	# define fap function
	connection.create_function("fap", 2, lambda rank, ifostr: Far.fap_from_rank(rank, lsctables.instrument_set_from_ifos(ifostr)))

	# FIXME abusing false_alarm_rate column, move for a false_alarm_probability column??
	connection.cursor().execute("UPDATE coinc_inspiral SET false_alarm_rate = (SELECT fap(coinc_event.likelihood, coinc_inspiral.ifos) FROM coinc_event WHERE coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id)")

	# all finished
	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = verbose)


#
# Function to compute the far in a given file
#


def set_far(Far, f, tmp_path = None, scale = None, verbose = False):
	working_filename = dbtables.get_connection_filename(f, tmp_path = tmp_path, verbose = verbose)
	connection = sqlite3.connect(working_filename)

	connection.create_function("far", 2, lambda rank, ifostr: Far.far_from_rank(rank, lsctables.instrument_set_from_ifos(ifostr), scale = scale))
	connection.cursor().execute('UPDATE coinc_inspiral SET combined_far = (SELECT far(coinc_event.likelihood, coinc_inspiral.ifos) FROM coinc_event WHERE coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id)')
	connection.commit()
	connection.close()

	dbtables.put_connection_filename(f, working_filename, verbose = verbose)


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
		progressbar.next(delta = n_burn)
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
			progressbar.next()
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
		progressbar.next(delta = progressbar.max)
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
