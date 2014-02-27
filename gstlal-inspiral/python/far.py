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
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments
from glue import segments
from glue.segmentsUtils import vote
from pylal import ligolw_burca_tailor
from pylal import ligolw_burca2
from pylal import inject
from pylal import progress
from pylal import rate


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


class ThincaCoincParamsDistributions(ligolw_burca_tailor.CoincParamsDistributions):
	# FIXME:  this is the name used for the burca distributions.
	# existing XML files were written using this name, so for
	# compatibility we stick to it for now, but in the future we should
	# pick a unique name so the files can't be confused for burca
	# files.
	ligo_lw_name_suffix = u"pylal_ligolw_burca_tailor_coincparamsdistributions"


#
# Paramter Distributions
#


class DistributionsStats(object):
	"""
	A class used to populate a ThincaCoincParamsDistribution instance using
	event parameter data.
	"""

	binnings = {
		"H1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(4., 100., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"H2_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(4., 100., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"H1H2_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(4., 100., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"L1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(4., 100., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"V1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(4., 100., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200)))
	}

	# FIXME the characteristic width (which is relevant for smoothing)
	# should be roughly 1.41 in SNR (from Gaussian noise expectations).  So
	# it is tied to how many bins there are per SNR range.  With 200 bins
	# between 4 and 26 each bin is .11 wide in SNR. So a width of 13 bins
	# corresponds to 1.43 which is close to 1.41
	filters = {
		"H1_snr_chi": rate.gaussian_window(7, 7, sigma = 10),
		"H2_snr_chi": rate.gaussian_window(7, 7, sigma = 10),
		"H1H2_snr_chi": rate.gaussian_window(7, 7, sigma = 10),
		"L1_snr_chi": rate.gaussian_window(7, 7, sigma = 10),
		"V1_snr_chi": rate.gaussian_window(7, 7, sigma = 10)
	}

	def __init__(self):
		self.raw_distributions = ThincaCoincParamsDistributions(**self.binnings)
		self.smoothed_distributions = ThincaCoincParamsDistributions(**self.binnings)
		self.likelihood_pdfs = {}
		self.target_length = 1000

	def __add__(self, other):
		out = type(self)()
		out.raw_distributions += self.raw_distributions
		out.raw_distributions += other.raw_distributions
		#FIXME do we also add the smoothed distributions??
		return out

	@staticmethod
	def likelihood_params_func(events, offsetvector):
		instruments = set(event.ifo for event in events)
		# don't allow both H1 and H2 to participate in the same
		# coinc.  if both have participated favour H1
		if "H1" in instruments:
			instruments.discard("H2")
		return dict(("%s_snr_chi" % event.ifo, (numpy.clip(event.snr, 0., 99.9), event.chisq / event.snr**2)) for event in events if event.ifo in instruments)

	def add_single(self, event):
		self.raw_distributions.add_background(self.likelihood_params_func((event,), None))

	def add_background_prior(self, n = 1., transition = 10., instruments = None, prefactors_range = (1.0, 10.0), df = 40, verbose = False):
		# Make a new empty coinc param distributions
		newdist = ThincaCoincParamsDistributions(**self.binnings)
		for param, binarr in newdist.background_rates.items():
			instrument = param.split("_")[0]
			# save some computation if we only requested certain instruments
			if instruments is not None and instrument not in instruments:
				continue
			# Custom handle the first and last over flow bins
			snrs = binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for snr in snrs:
				p = math.exp(-snr**2 / 2. + snrs[0]**2 / 2. + math.log(n))
				p += (transition / snr)**6 * math.exp(-transition**2 / 2. + snrs[0]**2 / 2. + math.log(n)) # Softer fall off above some transition SNR for numerical reasons
				for chi2_over_snr2 in chi2_over_snr2s:
					binarr[snr, chi2_over_snr2] += p
			# normalize to the requested count
			binarr.array /= binarr.array.sum()
			binarr.array *= n
		self.raw_distributions += newdist

		# FIXME, an adhoc way of adding glitches, us a signal distribution with bad matches
		# FIXME if we keep this, make a function that can be reused for both signal and background
		# FIXME:  for maintainability, this should be modified to
		# use the .add_injection() method of the .raw_distributions
		# attribute, but that will slow this down
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 10)
		# Make a new empty coinc param distributions
		newdist = ThincaCoincParamsDistributions(**self.binnings)
		for param, binarr in newdist.background_rates.items():
			instrument = param.split("_")[0]
			# save some computation if we only requested certain instruments
			if instruments is not None and instrument not in instruments:
				continue
			if verbose:
				print >> sys.stderr, "synthesizing background for %s" % param
			# Custom handle the first and last over flow bins
			snrs = binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for i, snr in enumerate(snrs):
				for j, chi2_over_snr2 in enumerate(chi2_over_snr2s):
					chisq = chi2_over_snr2 * snr**2 * df # We record the reduced chi2
					dist = 0
					for pf in pfs:
						nc = pf * snr**2
						v = stats.ncx2.pdf(chisq, df, nc)
						if numpy.isfinite(v):
							dist += v
					dist *= (snr / snrs[0])**-4
					if numpy.isfinite(dist):
						binarr[snr, chi2_over_snr2] += dist
			# normalize to the requested count
			binarr.array /= binarr.array.sum()
			binarr.array *= n
		self.raw_distributions += newdist

	def add_foreground_prior(self, n = 1., prefactors_range = (0.0, 0.10), df = 40, instruments = None, verbose = False):
		# FIXME:  for maintainability, this should be modified to
		# use the .add_injection() method of the .raw_distributions
		# attribute, but that will slow this down
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 10)
		# Make a new empty coinc param distributions
		newdist = ThincaCoincParamsDistributions(**self.binnings)
		for param, binarr in newdist.injection_rates.items():
			instrument = param.split("_")[0]
			# save some computation if we only requested certain instruments
			if instruments is not None and instrument not in instruments:
				continue
			if verbose:
				print >> sys.stderr, "synthesizing injections for %s" % param
			# Custom handle the first and last over flow bins
			snrs = binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for i, snr in enumerate(snrs):
				for j, chi2_over_snr2 in enumerate(chi2_over_snr2s):
					chisq = chi2_over_snr2 * snr**2 * df # We record the reduced chi2
					dist = 0
					for pf in pfs:
						nc = pf * snr**2
						v = stats.ncx2.pdf(chisq, df, nc)
						if numpy.isfinite(v):
							dist += v
					dist *= (snr / snrs[0])**-4
					if numpy.isfinite(dist):
						binarr[snr, chi2_over_snr2] += dist
			# normalize to the requested count
			binarr.array /= binarr.array.sum()
			binarr.array *= n
		self.raw_distributions += newdist

	def finish(self, verbose = False):
		self.smoothed_distributions = self.raw_distributions.copy(self.raw_distributions)
		#self.smoothed_distributions.finish(filters = self.filters, verbose = verbose)
		# FIXME:  should be the line above, we'll temporarily do
		# the following.  the difference is that the above produces
		# PDFs while what follows produces probabilities in each
		# bin
		if verbose:
			print >>sys.stderr, "smoothing parameter distributions ...",
		for name, binnedarray in itertools.chain(self.smoothed_distributions.background_rates.items(), self.smoothed_distributions.injection_rates.items()):
			if verbose:
				print >>sys.stderr, "%s," % name,
			rate.filter_array(binnedarray.array, self.filters[name])
			numpy.clip(binnedarray.array, 0.0, PosInf, binnedarray.array)
			binnedarray.array /= binnedarray.array.sum()
		if verbose:
			print >>sys.stderr, "done"

	def compute_single_instrument_background(self, instruments = None, verbose = False):
		# initialize a likelihood ratio evaluator
		likelihood_ratio_evaluator = ligolw_burca2.LikelihoodRatio(self.smoothed_distributions)

		# reduce typing
		background = self.smoothed_distributions.background_rates
		injections = self.smoothed_distributions.injection_rates

		self.likelihood_pdfs.clear()
		for param in background:
			# FIXME only works if there is a 1-1 relationship between params and instruments
			instrument = param.split("_")[0]
			
			# save some computation if we only requested certain instruments
			if instruments is not None and instrument not in instruments:
				continue

			if verbose:
				print >>sys.stderr, "updating likelihood background for %s" % instrument

			likelihoods = injections[param].array / background[param].array
			# ignore infs and nans because background is never
			# found in those bins.  the boolean array indexing
			# flattens the array
			minlikelihood, maxlikelihood = likelihood_bin_boundaries(likelihoods, background[param].array)
			# construct PDF
			# FIXME:  because the background array contains
			# probabilities and not probability densities, the
			# likelihood_pdfs contain probabilities and not
			# densities, as well, when this is done
			self.likelihood_pdfs[instrument] = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, self.target_length),)))
			for coords in iterutils.MultiIter(*background[param].bins.centres()):
				likelihood = likelihood_ratio_evaluator({param: coords})
				if numpy.isfinite(likelihood):
					self.likelihood_pdfs[instrument][likelihood,] += background[param][coords]

	@classmethod
	def from_xml(cls, xml, name):
		self = cls()
		self.raw_distributions, process_id = ThincaCoincParamsDistributions.from_xml(xml, name)
		# FIXME:  produce error if binnings don't match this class's binnings attribute?
		binnings = dict((param, self.raw_distributions.zero_lag_rates[param].bins) for param in self.raw_distributions.zero_lag_rates)
		self.smoothed_distributions = ThincaCoincParamsDistributions(**binnings)
		return self, process_id

	@classmethod
	def from_filenames(cls, filenames, contenthandler = DefaultContentHandler, verbose = False):
		self = cls()
		self.raw_distributions, seglists = ligolw_burca_tailor.load_likelihood_data(filenames, ThincaCoincParamsDistributions, u"gstlal_inspiral_likelihood", contenthandler = contenthandler, verbose = verbose)
		# FIXME:  produce error if binnings don't match this class's binnings attribute?
		binnings = dict((param, self.raw_distributions.zero_lag_rates[param].bins) for param in self.raw_distributions.zero_lag_rates)
		self.smoothed_distributions = ThincaCoincParamsDistributions(**binnings)
		return self, seglists

	def to_xml(self, process, name):
		return self.raw_distributions.to_xml(process, name)


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


def likelihood_bin_boundaries(likelihoods, probabilities, minint = 1e-2, maxint = (1 - 1e-14)):
	"""
	A function to choose the likelihood bin boundaries based on a certain
	interval in the likelihood pdfs set by minint and maxint. This should typically
	be combined with Overflow binning to catch the edges
	"""
	finite_likelihoods = numpy.isfinite(likelihoods)
	likelihoods = likelihoods[finite_likelihoods]
	background_pdf = probabilities[finite_likelihoods]
	
	sortindex = likelihoods.argsort()
	likelihoods = likelihoods[sortindex]
	background_pdf = background_pdf[sortindex]

	s = background_pdf.cumsum() / background_pdf.sum()
	# restrict range to a reasonable confidence interval to make the best use of our bin resolution
	minlikelihood = likelihoods[s.searchsorted(minint, side = 'right')]
	maxlikelihood = likelihoods[s.searchsorted(maxint)]
	if minlikelihood == 0:
		minlikelihood = max(likelihoods[likelihoods != 0].min(), 1e-100) # to prevent numerical issues
	return minlikelihood, maxlikelihood


def possible_ranks_array(likelihood_pdfs, ifo_set, targetlen):
	# start with an identity array to seed the outerproduct chain
	ranks = numpy.array([1.0])
	vals = numpy.array([1.0])
	# FIXME:  probably only works because the pdfs aren't pdfs but probabilities
	for ifo in ifo_set:
		likelihood_pdf = likelihood_pdfs[ifo]
		# FIXME lower instead of centres() to avoid inf in the last bin
		ranks = numpy.outer(ranks, likelihood_pdf.bins.lower()[0])
		vals = numpy.outer(vals, likelihood_pdf.array)
		ranks = ranks.reshape((ranks.shape[0] * ranks.shape[1],))
		vals = vals.reshape((vals.shape[0] * vals.shape[1],))
		# rebin the outer-product
		minlikelihood, maxlikelihood = likelihood_bin_boundaries(ranks, vals)
		new_likelihood_pdf = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, targetlen),)))
		for rank,val in zip(ranks,vals):
			new_likelihood_pdf[rank,] += val
		ranks = new_likelihood_pdf.bins.lower()[0]
		vals = new_likelihood_pdf.array

	# FIXME the size of these is targetlen which has to be small
	# since it is squared in the outer product.  We can afford to
	# store a 1e6 array.  Maybe we should try to make the resulting
	# pdf bigger somehow?
	return new_likelihood_pdf


#
# Class to handle the computation of FAPs/FARs
#


class LocalRankingData(object):
	def __init__(self, livetime_seg, distribution_stats, trials_table = None, target_length = 1000):
		self.distribution_stats = distribution_stats
		if trials_table is None:
			self.trials_table = TrialsTable()
		else:
			self.trials_table = trials_table
		self.joint_likelihood_pdfs = {}
		self.livetime_seg = livetime_seg

		#
		# the target FAP resolution is 1000 bins by default. This is purely
		# for memory/CPU requirements
		#

		self.target_length = target_length

	def __iadd__(self, other):
		self.distribution_stats += other.distribution_stats
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

		return self

	@classmethod
	def from_xml(cls, xml, name = u"gstlal_inspiral_likelihood"):
		llw_elem, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_FAR" % name]
		distribution_stats, process_id = DistributionsStats.from_xml(llw_elem, name)
		# the code that writes these things has put the
		# livetime_seg into the out segment in the search_summary
		# table.  uninitialized segments got recorded as
		# [None,None)
		livetime_seg, = (row.get_out() for row in lsctables.table.get_table(xml, lsctables.SearchSummaryTable.tableName) if row.process_id == process_id)
		self = cls(livetime_seg, distribution_stats = distribution_stats, trials_table = TrialsTable.from_xml(llw_elem))

		# pull out the joint likelihood arrays if they are present
		for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and "_joint_likelihood" in elem.getAttribute(u"Name")]:
			ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.getAttribute(u"Name").split("_")[0]))
			self.joint_likelihood_pdfs[ifo_set] = rate.binned_array_from_xml(ba_elem, ba_elem.getAttribute(u"Name").split(":")[0])
		
		return self, process_id

	@classmethod
	def from_filenames(cls, filenames, name = u"gstlal_inspiral_likelihood", contenthandler = DefaultContentHandler, verbose = False):
		self, process_id = LocalRankingData.from_xml(ligolw_utils.load_filename(filenames[0], contenthandler = contenthandler, verbose = verbose), name = name)
		for f in filenames[1:]:
			s, p = LocalRankingData.from_xml(ligolw_utils.load_filename(f, contenthandler = contenthandler, verbose = verbose), name = name)
			self += s
		return self
		
	def to_xml(self, process, name = u"gstlal_inspiral_likelihood"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_FAR" % name})
		xml.appendChild(self.trials_table.to_xml())
		xml.appendChild(self.distribution_stats.to_xml(process, name))
		for key in self.joint_likelihood_pdfs:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			xml.appendChild(rate.binned_array_to_xml(self.joint_likelihood_pdfs[key], "%s_joint_likelihood" % (ifostr,)))
		return xml

	def smooth_distribution_stats(self, verbose = False):
		if self.distribution_stats is not None:
			# FIXME:  this results in the
			# .smoothed_distributions object containing
			# *probabilities* not probability densities. this
			# might be changed in the future.
			self.distribution_stats.finish(verbose = verbose)

	def compute_joint_instrument_background(self, remap, instruments = None, verbose = False):
		# first get the single detector distributions
		self.distribution_stats.compute_single_instrument_background(instruments = instruments, verbose = verbose)

		self.joint_likelihood_pdfs.clear()

		# calculate all of the possible ifo combinations with at least
		# 2 detectors in order to get the joint likelihood pdfs
		likelihood_pdfs = self.distribution_stats.likelihood_pdfs
		if instruments is None:
			instruments = likelihood_pdfs.keys()
		for n in range(2, len(instruments) + 1):
			for ifos in iterutils.choices(instruments, n):
				ifo_set = frozenset(ifos)
				remap_set = remap.setdefault(ifo_set, ifo_set)

				if verbose:
					print >>sys.stderr, "computing joint likelihood background for %s remapped to %s" % (lsctables.ifos_from_instrument_set(ifo_set), lsctables.ifos_from_instrument_set(remap_set))

				# only recompute if necessary, some choices
				# may not be if a certain remap set is
				# provided
				if remap_set not in self.joint_likelihood_pdfs:
					self.joint_likelihood_pdfs[remap_set] = possible_ranks_array(likelihood_pdfs, remap_set, self.target_length)

				self.joint_likelihood_pdfs[ifo_set] = self.joint_likelihood_pdfs[remap_set]


class RankingData(object):
	def __init__(self, local_ranking_data):
		# ensure the trials tables' keys match the likelihood
		# histograms' keys
		assert set(local_ranking_data.joint_likelihood_pdfs) == set(local_ranking_data.trials_table)

		# copy likelihood ratio PDFs
		self.joint_likelihood_pdfs = dict((key, copy.deepcopy(value)) for key, value in local_ranking_data.joint_likelihood_pdfs.items())

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
		fake_local_ranking_data.joint_likelihood_pdfs = {}
		for key in fake_local_ranking_data.trials_table:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			fake_local_ranking_data.joint_likelihood_pdfs[key] = rate.binned_array_from_xml(llw_elem, ifostr)

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
		for key in self.joint_likelihood_pdfs:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			xml.appendChild(rate.binned_array_to_xml(self.joint_likelihood_pdfs[key], ifostr))
		assert search_summary.process_id == process.process_id
		search_summary.set_out(self.livetime_seg)
		return xml


	def __iadd__(self, other):
		our_trials = self.trials_table
		other_trials = other.trials_table

		our_keys = set(self.joint_likelihood_pdfs)
		other_keys  = set(other.joint_likelihood_pdfs)

		# PDFs that only we have are unmodified
		pass

		# PDFs that only the new data has get copied verbatim
		for k in other_keys - our_keys:
			self.joint_likelihood_pdfs[k] = copy.deepcopy(other.joint_likelihood_pdfs[k])

		# PDFs that we have and are in the new data get replaced
		# with the weighted sum, re-binned
		for k in our_keys & other_keys:
			minself, maxself, nself = self.joint_likelihood_pdfs[k].bins[0].min, self.joint_likelihood_pdfs[k].bins[0].max, self.joint_likelihood_pdfs[k].bins[0].n
			minother, maxother, nother = other.joint_likelihood_pdfs[k].bins[0].min, other.joint_likelihood_pdfs[k].bins[0].max, other.joint_likelihood_pdfs[k].bins[0].n
			new_joint_likelihood_pdf =  rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(min(minself, minother), max(maxself, maxother), max(nself, nother)),)))

			for x in self.joint_likelihood_pdfs[k].centres()[0]:
				new_joint_likelihood_pdf[x,] += self.joint_likelihood_pdfs[k][x,] * float(our_trials[k].count or 1) / ((our_trials[k].count + other_trials[k].count) or 1)
			for x in other.joint_likelihood_pdfs[k].centres()[0]:
				new_joint_likelihood_pdf[x,] += other.joint_likelihood_pdfs[k][x,] * float(other_trials[k].count or 1) / ((our_trials[k].count + other_trials[k].count) or 1)

			self.joint_likelihood_pdfs[k] = new_joint_likelihood_pdf

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
		for key, lpdf in self.joint_likelihood_pdfs.items():
			ranks = lpdf.bins.lower()[0]
			weights = lpdf.array
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
			self.cdf_interpolator[key] = interpolate.interp1d(ranks, cdf)
			self.ccdf_interpolator[key] = interpolate.interp1d(ranks, ccdf)
			# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
			self.minrank[key] = min(ranks)
			self.maxrank[key] = max(ranks)

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
