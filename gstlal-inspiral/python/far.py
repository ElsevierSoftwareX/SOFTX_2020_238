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


import bisect
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
import warnings
from scipy import interpolate
from scipy import optimize
from scipy import stats
import sqlite3
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
from glue.segmentsUtils import vote
from glue.text_progress_bar import ProgressBar
from gstlal import emcee
from pylal import inject
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
# Horizon distance record keeping
#


class NearestLeafTree(object):
	"""
	A simple binary tree in which look-ups return the value of the
	closest leaf.  Only float objects are supported for the keys and
	values.  Look-ups raise KeyError if the tree is empty.

	Example:

	>>> x = NearestLeafTree()
	>>> x[100.0] = 120.
	>>> x[104.0] = 100.
	>>> x[102.0] = 110.
	>>> x[90.]
	120.0
	>>> x[100.999]
	120.0
	>>> x[101.001]
	110.0
	>>> x[200.]
	100.0
	>>> del x[104]
	>>> x[200.]
	110.0
	>>> x.keys()
	[100.0, 102.0]
	>>> 102 in x
	True
	>>> 103 in x
	False
	>>> x.to_xml(u"H1").write()
	<Array Type="real_8" Name="H1:nearestleaftree:array">
		<Dim>2</Dim>
		<Dim>2</Dim>
		<Stream Delimiter=" " Type="Local">
			100 102
			120 110
		</Stream>
	</Array>
	"""
	def __init__(self, items = ()):
		"""
		Initialize a NearestLeafTree.

		Example:

		>>> x = NearestLeafTree()
		>>> x = NearestLeafTree([(100., 120.), (104., 100.), (102., 110.)])
		>>> y = {100.: 120., 104.: 100., 102.: 100.}
		>>> x = NearestLeafTree(y.items())
		"""
		self.tree = list(items)
		self.tree.sort()

	def __setitem__(self, x, val):
		"""
		Example:

		>>> x = NearestLeafTree()
		>>> x[100.:200.] = 0.
		>>> x[150.] = 1.
		>>> x
		NearestLeaftree([(100, 0), (150, 1), (200, 0)])
		"""
		if type(x) is slice:
			# replace all entries in the requested range of
			# co-ordiantes with two entries, each with the
			# given value, one at the start of the range and
			# one at the end of the range.  thus, after this
			# all queries within that range will return this
			# value.
			if x.step is not None:
				raise ValueError("%s: step not supported" % repr(x))
			if x.start is None:
				if not self.tree:
					raise IndexError("open-ended slice not supported with empty tree")
				x = slice(self.minkey(), x.stop)
			if x.stop is None:
				if not self.tree:
					raise IndexError("open-ended slice not supported with empty tree")
				x = slice(x.start, self.maxkey())
			if x.stop < x.start:
				raise ValueError("%s: bounds out of order" % repr(x))
			lo = bisect.bisect_left(self.tree, (x.start, NegInf))
			hi = bisect.bisect_right(self.tree, (x.stop, PosInf))
			self.tree[lo:hi] = ((x.start, val), (x.stop, val))
		else:
			# replace all entries having the same co-ordinate
			# with this one
			lo = bisect.bisect_left(self.tree, (x, NegInf))
			hi = bisect.bisect_right(self.tree, (x, PosInf))
			self.tree[lo:hi] = ((x, val),)

	def __getitem__(self, x):
		if not self.tree:
			raise KeyError(x)
		if type(x) is slice:
			raise ValueError("slices not supported")
		hi = bisect.bisect_right(self.tree, (x, PosInf))
		try:
			x_hi, val_hi = self.tree[hi]
		except IndexError:
			x_hi, val_hi = self.tree[-1]
		if hi == 0:
			x_lo, val_lo = x_hi, val_hi
		else:
			x_lo, val_lo = self.tree[hi - 1]
		# compute average in way that will be safe if x values are
		# range-limited objects
		return val_lo if x < x_lo + (x_hi - x_lo) / 2. else val_hi

	def __delitem__(self, x):
		"""
		Example:

		>>> x = NearestLeafTree([(100., 0.), (150., 1.), (200., 0.)])
		>>> del x[150.]
		>>> x
		NearestLeafTree([(100., 0.), (200., 0.)])
		>>> del x[:]
		NearestLeafTree([])
		"""
		if type(x) is slice:
			if x.step is not None:
				raise ValueError("%s: step not supported" % repr(x))
			if x.start is None:
				if not self.tree:
					# no-op
					return
				x = slice(self.minkey(), x.stop)
			if x.stop is None:
				if not self.tree:
					# no-op
					return
				x = slice(x.start, self.maxkey())
			if x.stop < x.start:
				# no-op
				return
			lo = bisect.bisect_left(self.tree, (x.start, NegInf))
			hi = bisect.bisect_right(self.tree, (x.stop, PosInf))
			del self.tree[lo:hi]
		elif not self.tree:
			raise IndexError(x)
		else:
			lo = bisect.bisect_left(self.tree, (x, NegInf))
			if self.tree[lo][0] != x:
				raise IndexError(x)
			del self.tree[lo]

	def __nonzero__(self):
		return bool(self.tree)

	def __iadd__(self, other):
		for x, val in other.tree:
			self[x] = val
		return self

	def keys(self):
		return [x for x, val in self.tree]

	def values(self):
		return [val for x, val in self.tree]

	def items(self):
		return list(self.tree)

	def min(self):
		"""
		Return the minimum value stored in the tree.  This is O(n).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return min(val for x, val in self.tree)

	def minkey(self):
		"""
		Return the minimum key stored in the tree.  This is O(1).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return self.tree[0][0]

	def max(self):
		"""
		Return the maximum value stored in the tree.  This is O(n).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return max(val for x, val in self.tree)

	def maxkey(self):
		"""
		Return the maximum key stored in the tree.  This is O(1).
		"""
		if not self.tree:
			raise ValueError("empty tree")
		return self.tree[-1][0]

	def __contains__(self, x):
		try:
			return bool(self.tree) and self.tree[bisect.bisect_left(self.tree, (x, NegInf))][0] == x
		except IndexError:
			return False

	def __len__(self):
		return len(self.tree)

	def __repr__(self):
		return "NearestLeaftree([%s])" % ", ".join("(%g, %g)" % item for item in self.tree)

	@classmethod
	def from_xml(cls, xml, name):
		return cls(map(tuple, ligolw_array.get_array(xml, u"%s:nearestleaftree" % name).array[:]))

	def to_xml(self, name):
		return ligolw_array.from_array(u"%s:nearestleaftree" % name, numpy.array(self.tree, dtype = "double"))


class HorizonHistories(dict):
	def __iadd__(self, other):
		for key, history in other.iteritems():
			try:
				self[key] += history
			except KeyError:
				self[key] = copy.deepcopy(history)
		return self

	def minkey(self):
		"""
		Return the minimum key stored in the trees.
		"""
		minkeys = tuple(history.minkey() for history in self.values() if history)
		if not minkeys:
			raise ValueError("empty trees")
		return min(minkeys)

	def maxkey(self):
		"""
		Return the maximum key stored in the trees.
		"""
		maxkeys = tuple(history.maxkey() for history in self.values() if history)
		if not maxkeys:
			raise ValueError("empty trees")
		return max(maxkeys)

	def getdict(self, x):
		return dict((key, value[x]) for key, value in self.iteritems())

	def randhorizons(self):
		"""
		Generator yielding a sequence of random horizon distance
		dictionaries chosen by drawing random times uniformly
		distributed between the lowest and highest times recorded
		in the history and returning the dictionary of horizon
		distances for each of those times.
		"""
		x_min = self.minkey()
		x_max = self.maxkey()
		getdict = self.getdict
		rnd = random.uniform
		while 1:
			yield getdict(rnd(x_min, x_max))

	def all(self):
		"""
		Returns a list of the unique sets of horizon distances
		recorded in the histories.
		"""
		# unique times for which a horizon distance measurement is
		# available
		all_x = set(x for value in self.values() for x in value.keys())

		# the unique horizon distances from those times, expressed
		# as frozensets of instrument/distance pairs
		result = set(frozenset(self.getdict(x).items()) for x in all_x)

		# return a list of the results converted back to
		# dictionaries
		return map(dict, result)

	@classmethod
	def from_xml(cls, xml, name):
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == u"%s:horizonhistories" % name]
		try:
			xml, = xml
		except ValueError:
			raise ValueError("document must contain exactly 1 HorizonHistories named '%s'" % name)
		keys = [elem.Name.replace(u":nearestleaftree", u"") for elem in xml.getElementsByTagName(ligolw.Array.tagName) if elem.hasAttribute(u"Name") and elem.Name.endswith(u":nearestleaftree")]
		self = cls()
		for key in keys:
			self[key] = NearestLeafTree.from_xml(xml, key)
		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:horizonhistories" % name})
		for key, value in self.items():
			xml.appendChild(value.to_xml(key))
		return xml


#
# Inspiral-specific CoincParamsDistributions sub-class
#


class ThincaCoincParamsDistributions(snglcoinc.CoincParamsDistributions):
	ligo_lw_name_suffix = u"gstlal_inspiral_coincparamsdistributions"

	instrument_categories = snglcoinc.InstrumentCategories()

	# range of SNRs covered by this object
	# FIXME:  must ensure lower boundary matches search threshold
	snr_min = 4.

	# if two horizon distances, D1 and D2, differ by less than
	#
	#	| ln(D1 / D2) | <= log_distance_tolerance
	#
	# then they are considered to be equal for the purpose of recording
	# horizon distance history, generating joint SNR PDFs, and so on.
	#
	# FIXME:  is this choice of distance quantization appropriate?
	log_distance_tolerance = math.log(1.2)

	# binnings (filter funcs look-up initialized in .__init__()
	binnings = {
		"instruments": rate.NDBins((rate.LinearBins(0.5, instrument_categories.max() + 0.5, instrument_categories.max()),)),
		"H1_snr_chi": rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 260), rate.ATanLogarithmicBins(.001, 0.5, 200))),
		"H2_snr_chi": rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 260), rate.ATanLogarithmicBins(.001, 0.5, 200))),
		"H1H2_snr_chi": rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 260), rate.ATanLogarithmicBins(.001, 0.5, 200))),
		"L1_snr_chi": rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 260), rate.ATanLogarithmicBins(.001, 0.5, 200))),
		"V1_snr_chi": rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 260), rate.ATanLogarithmicBins(.001, 0.5, 200)))
	}

	def __init__(self, *args, **kwargs):
		super(ThincaCoincParamsDistributions, self).__init__(*args, **kwargs)
		self.horizon_history = HorizonHistories()
		self.pdf_from_rates_func = {
			"instruments": self.pdf_from_rates_instruments,
			"H1_snr_chi": self.pdf_from_rates_snrchi2,
			"H2_snr_chi": self.pdf_from_rates_snrchi2,
			"H1H2_snr_chi": self.pdf_from_rates_snrchi2,
			"L1_snr_chi": self.pdf_from_rates_snrchi2,
			"V1_snr_chi": self.pdf_from_rates_snrchi2,
		}

	def __iadd__(self, other):
		# NOTE:  because we use custom PDF constructions, the stock
		# .__iadd__() method for this class will not result in
		# valid PDFs.  the rates arrays *are* handled correctly by
		# the .__iadd__() method, by fiat, so just remember to
		# invoke .finish() to get the PDFs in shape afterwards
		super(ThincaCoincParamsDistributions, self).__iadd__(other)
		self.horizon_history += other.horizon_history
		return self

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
	# 5**3 * 4 = 5 quantizations in 3 instruments, 4 combos per
	# quantized choice of horizon distances
	#

	max_cached_snr_joint_pdfs = int(5**3 * 4)
	snr_joint_pdf_cache = {}

	def get_snr_joint_pdf(self, instruments, horizon_distances, verbose = False):
		#
		# key for cache:  two element tuple, first element is
		# frozen set of instruments for which this is the PDF,
		# second element is frozen set of (instrument, horizon
		# distance) pairs for all instruments in the network.
		# horizon distances are normalized to fractions of the
		# largest among them and then are quantized to integer
		# powers of exp(log_distance_tolerance)
		#
		# FIXME:  if horizon distance discrepancy is too large,
		# consider a fast-path that just returns an all-0 array
		#

		horizon_distance_norm = max(horizon_distances.values())
		key = frozenset(instruments), frozenset((instrument, math.exp(round(math.log(horizon_distance / horizon_distance_norm) / self.log_distance_tolerance) * self.log_distance_tolerance)) for instrument, horizon_distance in horizon_distances.items())

		#
		# retrieve cached PDF, or build new one
		#

		try:
			pdf = self.snr_joint_pdf_cache[key][0]
		except KeyError:
			# no entries in cache for this instrument combo and
			# set of horizon distances
			if self.snr_joint_pdf_cache:
				age = max(age for ignored, ignored, age in self.snr_joint_pdf_cache.values()) + 1
			else:
				age = 0
			if verbose:
				print >>sys.stderr, "For horizon distances %s" % ", ".join("%s = %.4g Mpc" % item for item in sorted(horizon_distances.items()))
				progressbar = ProgressBar(text = "%s SNR PDF" % ", ".join(sorted(key[0])))
			else:
				progressbar = None
			binnedarray = self.joint_pdf_of_snrs(key[0], dict(key[1]), progressbar = progressbar)
			pdf = rate.InterpBinnedArray(binnedarray)
			self.snr_joint_pdf_cache[key] = pdf, binnedarray, age
			# if the cache is full, delete the entry with the
			# smallest age
			while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
				del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
			del progressbar
		return pdf

	def coinc_params(self, events, offsetvector):
		params = dict(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events)
		# don't allow both H1 and H2 to participate in the same
		# coinc.  if both have participated favour H1
		if "H2_snr_chi" in params and "H1_snr_chi" in params:
			del params["H2_snr_chi"]
		params["instruments"] = (ThincaCoincParamsDistributions.instrument_categories.category(event.ifo for event in events),)

		# pick one trigger at random to provide a timestamp and
		# pull the horizon distances from our horizon distance
		# history at that time.  the horizon history is keyed by
		# floating-point values (don't need nanosecond precision
		# for this)
		horizons = self.horizon_history.getdict(float(events[0].get_end()))
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
		#horizons.update(dict((event.ifo, event.eff_distance * event.snr / 8.) for event in events))

		params["horizons"] = horizons

		return params

	def P_noise(self, params):
		if params is not None:
			params = params.copy()
			del params["horizons"]
		return super(ThincaCoincParamsDistributions, self).P_noise(params)

	def P_signal(self, params):
		if params is None:
			return None
		# (instrument, snr) pairs sorted alphabetically by instrument name
		snrs = sorted((name.split("_")[0], value[0]) for name, value in params.items() if name.endswith("_snr_chi"))
		# retrieve the SNR PDF
		snr_pdf = self.get_snr_joint_pdf((instrument for instrument, rho in snrs), params["horizons"])
		# evaluate it (snrs are alphabetical by instrument)
		P = snr_pdf(*tuple(rho for instrument, rho in snrs))

		# FIXME:  P(instruments | signal) needs to depend on
		# horizon distances.  here we're assuming whatever
		# populate_prob_of_instruments_given_signal() has set the probabilities to is
		# OK.  we probably need to cache these and save them in the
		# XML file, too, like P(snrs | signal, instruments)
		for name, value in params.items():
			if name != "horizons":
				P *= self.injection_pdf_interp[name](*value)
		return P

	# FIXME:  this is annoying.  probably need a generic way to
	# indicate to the parent class that some parameter doesn't have a
	# corresponding rate array
	def add_zero_lag(self, params, *args, **kwargs):
		if params is not None:
			params = params.copy()
			del params["horizons"]
		return super(ThincaCoincParamsDistributions, self).add_zero_lag(params, *args, **kwargs)
	def add_injection(self, params, *args, **kwargs):
		if params is not None:
			params = params.copy()
			del params["horizons"]
		return super(ThincaCoincParamsDistributions, self).add_injection(params, *args, **kwargs)
	def add_background(self, params, *args, **kwargs):
		if params is not None:
			params = params.copy()
			del params["horizons"]
		return super(ThincaCoincParamsDistributions, self).add_background(params, *args, **kwargs)

	def add_background_prior(self, instruments, n = 1., transition = 23., verbose = False):
		#
		# populate snr,chi2 binnings with a slope to force
		# higher-SNR events to be assesed to be more significant
		# when in the regime beyond the edge of measured or even
		# extrapolated background.
		#

		if verbose:
			print >>sys.stderr, "adding tilt to (SNR, \\chi^2) background PDFs ..."
		for instrument in instruments:
			binarr = self.background_rates["%s_snr_chi" % instrument]
			if verbose:
				progressbar = ProgressBar(instrument, max = len(binarr.bins[0]))
			else:
				progressbar = None

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)
			# don't compute this in the loop
			transition_factor = transition**5. * math.exp(-transition**2. / 2.)
			# iterate over all bins
			dchi2_over_snr2s = new_binarr.bins[1].upper() - new_binarr.bins[1].lower()
			dchi2_over_snr2s[numpy.isinf(dchi2_over_snr2s)] = 0.
			for snr, dsnr in zip(new_binarr.bins[0].centres(), new_binarr.bins[0].upper() - new_binarr.bins[0].lower()):
				# normalization is irrelevant.  overall
				# normalization will be imposed afterwards.
				# tilt looks like expected behaviour for
				# Gaussian noise + softer fall-off above
				# some transition SNR to avoid zero-valued
				# bins
				#
				# NOTE:  expression split up so that if a
				# numerical overflow or underflow occurs we
				# can see which part of the expression was
				# the problem from the traceback
				if math.isinf(dsnr):
					continue
				p = math.exp(-snr**2. / 2.)
				p += transition_factor / snr**5.
				new_binarr[snr,:] += p * dsnr * dchi2_over_snr2s
				if progressbar is not None:
					progressbar.increment()
			# zero everything below the search threshold and
			# normalize what's left to the requested count
			new_binarr[:self.snr_min,:] = 0.
			new_binarr.array *= n / new_binarr.array.sum()
			# add to raw counts
			self.background_rates["instruments"][self.instrument_categories.category([instrument]),] += n
			binarr += new_binarr

	def add_instrument_combination_counts(self, segs, verbose = False):
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
			print >>sys.stderr, "synthesizing background-like instrument combination probabilities ..."
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
		N = 0
		coincidence_bins = 0.
		for instruments in coincsynth.all_instrument_combos:
			predicted_count = coincsynth.rates[frozenset(instruments)] * livetime
			observed_count = self.zero_lag_rates["instruments"][self.instrument_categories.category(instruments),]
			if predicted_count > 0 and observed_count > 0:
				coincidence_bins += (predicted_count / observed_count)**(1. / (len(instruments) - 1))
				N += 1
		#FIXME this code path should not exist. You should not be allowed to do this. But for now it is useful for creating "prior" information.
		if (coincidence_bins == 0) or (N == 0):
			warnings.warn("There seems to be insufficient information to compute coincidence rates, setting them to unity")
			for instruments, count in coincsynth.mean_coinc_count.items():
				self.background_rates["instruments"][self.instrument_categories.category(instruments),] = 1.
		else:
			coincidence_bins /= N
			if verbose:
				print >>sys.stderr, "\tthere seems to be %g effective disjoint coincidence bin(s)" % coincidence_bins
			assert coincidence_bins >= 1.
			# convert single-instrument event rates to rates/bin
			coincsynth.mu = dict((instrument, rate / coincidence_bins) for instrument, rate in coincsynth.mu.items())
			# now compute the expected coincidence rates/bin,
			# then multiply by the number of bins to get the
			# expected coincidence rates
			for instruments, count in coincsynth.mean_coinc_count.items():
				self.background_rates["instruments"][self.instrument_categories.category(instruments),] = count * coincidence_bins


	def add_foreground_snrchi_prior(self, target_dict, instruments, n, prefactors_range, df, verbose = False):
		if verbose:
			print >>sys.stderr, "synthesizing signal-like (SNR, \\chi^2) distributions ..."
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 10)
		for instrument in instruments:
			binarr = target_dict["%s_snr_chi" % instrument]
			if verbose:
				progressbar = ProgressBar(instrument, max = len(binarr.bins[0]))
			else:
				progressbar = None

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)

			# iterate over all bins
			chi2_over_snr2s = new_binarr.bins[1].centres()
			dchi2_over_snr2s = new_binarr.bins[1].upper() - new_binarr.bins[1].lower()
			dchi2_over_snr2s[numpy.isinf(dchi2_over_snr2s)] = 0.
			for snr, dsnr in zip(new_binarr.bins[0].centres(), new_binarr.bins[0].upper() - new_binarr.bins[0].lower()):
				if math.isinf(dsnr):
					continue
				snr2 = snr**2.	# don't compute in loops
				for chi2_over_snr2, dchi2_over_snr2 in zip(chi2_over_snr2s, dchi2_over_snr2s):
					chi2 = chi2_over_snr2 * snr2 * df # We record the reduced chi2
					with numpy.errstate(over = "ignore", divide = "ignore", invalid = "ignore"):
						v = stats.ncx2.pdf(chi2, df, pfs * snr)
					# remove nans and infs, and barf if
					# we got a negative number
					v = v[numpy.isfinite(v)]
					assert (v >= 0.).all(), v
					# normalization is irrelevant,
					# final result will have over-all
					# normalization imposed
					p = v.sum() * snr**-4. * dsnr * dchi2_over_snr2
					assert p >= 0. and not (math.isinf(p) or math.isnan(p)), p
					new_binarr[snr, chi2_over_snr2] = p
				if progressbar is not None:
					progressbar.increment()
			# zero everything below the search threshold and
			# normalize what's left to the requested count
			new_binarr[:self.snr_min,:] = 0.
			new_binarr.array *= n / new_binarr.array.sum()
			# add to raw counts
			binarr += new_binarr

	def populate_prob_of_instruments_given_signal(self, segs, n = 1., verbose = False):
		#
		# populate instrument combination binning
		#

		assert len(segs) > 1
		assert set(self.horizon_history) <= set(segs)

		# probability that a signal is detectable by each of the
		# instrument combinations
		P = P_instruments_given_signal(self.horizon_history)

		# multiply by probability that enough instruments are on to
		# form each of those combinations
		#
		# FIXME:  if when an instrument is off it has its horizon
		# distance set to 0 in the horizon history, then this step
		# will not be needed because the marginalization over
		# horizon histories will already reflect the duty cycles.
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

	def _rebuild_interpolators(self):
		super(ThincaCoincParamsDistributions, self)._rebuild_interpolators()

		#
		# the instrument combination "interpolators" are pass-throughs
		#

		self.background_pdf_interp["instruments"] = lambda x: self.background_pdf["instruments"][x,]
		self.injection_pdf_interp["instruments"] = lambda x: self.injection_pdf["instruments"][x,]
		self.zero_lag_pdf_interp["instruments"] = lambda x: self.zero_lag_pdf["instruments"][x,]

	def pdf_from_rates_instruments(self, key, pdf_dict):
		# instrument combos are probabilities, not densities.  be
		# sure the single-instrument categories are zeroed.
		binnedarray = pdf_dict[key]
		for category in self.instrument_categories.values():
			binnedarray[category,] = 0
		with numpy.errstate(invalid = "ignore"):
			binnedarray.array /= binnedarray.array.sum()

	def pdf_from_rates_snrchi2(self, key, pdf_dict, snr_kernel_width_at_8 = math.sqrt(2) / 4.0, sigma = 10.):
		# get the binned array we're going to process
		binnedarray = pdf_dict[key]

		# construct the kernel
		snr_bins = binnedarray.bins[0]
		snr_per_bin_at_8 = (snr_bins.upper() - snr_bins.lower())[snr_bins[8.]]
		snr_kernel_bins = snr_kernel_width_at_8 / snr_per_bin_at_8
		assert snr_kernel_bins >= 2.5, snr_kernel_bins	# don't let the window get too small
		kernel = rate.gaussian_window(snr_kernel_bins, 5, sigma = sigma)

		# smooth the bin count data
		rate.filter_array(binnedarray.array, kernel)

		# zero everything below the SNR cut-off
		binnedarray[:self.snr_min,:] = 0.

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

	@classmethod
	def from_xml(cls, xml, name):
		self = super(ThincaCoincParamsDistributions, cls).from_xml(xml, name)
		xml = self.get_xml_root(xml, name)
		self.horizon_history = HorizonHistories.from_xml(xml, name)
		prefix = u"cached_snr_joint_pdf"
		for elem in [elem for elem in xml.childNodes if elem.Name.startswith(u"%s:" % prefix)]:
			key = ligolw_param.get_pyvalue(elem, u"key").strip().split(u";")
			key = frozenset(lsctables.instrument_set_from_ifos(key[0].strip())), frozenset((inst.strip(), float(dist.strip())) for inst, dist in (inst_dist.strip().split(u"=") for inst_dist in key[1].strip().split(u",")))
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
		xml.appendChild(self.horizon_history.to_xml(name))
		prefix = u"cached_snr_joint_pdf"
		for key, (ignored, binnedarray, ignored) in self.snr_joint_pdf_cache.items():
			elem = xml.appendChild(rate.binned_array_to_xml(binnedarray, prefix))
			elem.appendChild(ligolw_param.new_param(u"key", u"lstring", "%s;%s" % (lsctables.ifos_from_instrument_set(key[0]), u",".join(u"%s=%.17g" % inst_dist for inst_dist in sorted(key[1])))))
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
		"""
		snr_slope = 0.5

		keys = tuple("%s_snr_chi" % instrument for instrument in instruments)
		base_params = {"instruments": (self.instrument_categories.category(instruments),)}
		horizongen = iter(self.horizon_history.randhorizons()).next
		# P(horizons) = 1/livetime
		log_P_horizons = -math.log(self.horizon_history.maxkey() - self.horizon_history.minkey())
		coordgens = tuple(iter(self.binnings[key].randcentre(ns = (snr_slope, 1.))).next for key in keys)
		while 1:
			seq = sum((coordgen() for coordgen in coordgens), ())
			params = dict(zip(keys, seq[0::2]))
			params["horizons"] = horizongen()
			params.update(base_params)
			# NOTE:  I think the result of this sum is, in
			# fact, correctly normalized, but nothing requires
			# it to be (only that it be correct up to an
			# unknown constant) and I've not checked that it is
			# so the documentation doesn't promise that it is.
			yield params, sum(seq[1::2], log_P_horizons)

	@classmethod
	def joint_pdf_of_snrs(cls, instruments, inst_horiz_mapping, n_samples = 80000, bins = rate.ATanLogarithmicBins(3.6, 120., 100), progressbar = None):
		"""
		Return a BinnedArray representing the joint probability
		density of measuring a set of SNRs from a network of
		instruments.  The inst_horiz_mapping is a dictionary
		mapping instrument name (e.g., "H1") to horizon distance
		(arbitrary units).  n_samples is the number of lines over
		which to calculate the density in the SNR space.  The axes
		of the PDF correspond to the instruments in alphabetical
		order.
		"""
		# get instrument names in alphabetical order
		instruments = sorted(instruments)
		# get horizon distances and responses in that same order
		DH_times_8 = 8. * numpy.array([inst_horiz_mapping[inst] for inst in instruments])
		resps = tuple(inject.cached_detector[inject.prefix_to_name[inst]].response for inst in instruments)

		# get horizon distances and responses of remaining
		# instruments (order doesn't matter as long as they're in
		# the same order)
		DH_times_8_other = 8. * numpy.array([dist for inst, dist in inst_horiz_mapping.items() if inst not in instruments])
		resps_other = tuple(inject.cached_detector[inject.prefix_to_name[inst]].response for inst in inst_horiz_mapping if inst not in instruments)

		# initialize the PDF array, and pre-construct the sequence
		# of snr,d(snr) tuples.  since the last SNR bin probably
		# has infinite size, we remove it from the sequence
		# (meaning the PDF will be left 0 in that bin).
		pdf = rate.BinnedArray(rate.NDBins([bins] * len(instruments)))
		snr_sequence = rate.ATanLogarithmicBins(3.6, 120., 300)
		snr_snrlo_snrhi_sequence = numpy.array(zip(snr_sequence.centres(), snr_sequence.lower(), snr_sequence.upper())[:-1])

		# compute the SNR at which to begin iterations over bins
		assert type(cls.snr_min) is float
		snr_min = cls.snr_min - 3.0
		assert snr_min > 0.0

		# no-op if one of the instruments that must be able to
		# participate in a coinc is not on.  the PDF that gets
		# returned is not properly normalized (it's all 0) but
		# that's the correct value everywhere except at SNR-->+inf
		# where the product of SNR * no sensitivity might be said
		# to give a non-zero contribution, who knows.  anyway, for
		# finite SNR, 0 is the correct value.
		if DH_times_8.min() == 0.:
			return pdf

		# psi chooses the orientation of F+ and Fx, choosing a
		# fixed value is OK.  we select random
		# uniformly-distributed right ascensions so there's no
		# point in also choosing random GMSTs and any value is as
		# good as any other
		psi = gmst = 0.0

		# run the sampler the requested # of iterations.  save some
		# symbols to avoid doing module attribute look-ups in the
		# loop
		if progressbar is not None:
			progressbar.max = n_samples
		acos = math.acos
		random_uniform = random.uniform
		twopi = 2. * math.pi
		pi_2 = math.pi / 2.
		xlal_am_resp = inject.XLALComputeDetAMResponse
		for i in xrange(n_samples):
			theta = acos(random_uniform(-1., 1.))
			phi = random_uniform(0., twopi)
			cosi2 = random_uniform(-1., 1.)**2.

			# F+^2 and Fx^2 for each instrument
			fpfc2 = numpy.array(tuple(xlal_am_resp(resp, phi, pi_2 - theta, psi, gmst) for resp in resps))**2.
			fpfc2_other = numpy.array(tuple(xlal_am_resp(resp, phi, pi_2 - theta, psi, gmst) for resp in resps_other))**2.

			# ratio of distance to inverse SNR for each instrument
			snr_times_D = DH_times_8 * numpy.dot(fpfc2, ((1. + cosi2)**2. / 4., cosi2))**0.5

			# snr * D in instrument whose SNR grows fastest
			# with decreasing D
			max_snr_times_D = snr_times_D.max()

			# snr_times_D.min() / snr_min = the furthest a
			# source can be and still be above snr_min in all
			# instruments involved.  max_snr_times_D / that
			# distance = the SNR that distance corresponds to
			# in the instrument whose SNR grows fastest with
			# decreasing distance --- the SNR the source has in
			# the most sensitive instrument when visible to all
			# instruments in the combo
			min_D_at_snr_min = snr_times_D.min() / snr_min
			if min_D_at_snr_min == 0.:
				# one of the instruments that must be able
				# to see the event is blind to it (need
				# this check to avoid a divide-by-zero
				# error next)
				continue
			start_index = snr_sequence[max_snr_times_D / min_D_at_snr_min]

			# min_D_other is minimum distance at which source
			# becomes visible in an instrument that isn't
			# involved.  max_snr_times_D / min_D_other gives
			# the SNR in the most sensitive instrument at which
			# the source becomes visible to one of the
			# instruments not allowed to participate
			if len(DH_times_8_other):
				min_D_other = (DH_times_8_other * numpy.dot(fpfc2_other, ((1. + cosi2)**2. / 4., cosi2))**0.5).min() / cls.snr_min
				if min_D_other > 0.:
					end_index = snr_sequence[max_snr_times_D / min_D_other] + 1
				else:
					end_index = None
			else:
				end_index = None

			# if start_index >= end_index then in order for the
			# source to be close enough to be visible in all
			# the instruments that must see it it is already
			# visible to one or more instruments that must not.
			# don't need to check for this, the for loop that
			# comes next will simply not have any iterations.

			# iterate over the SNRs (= SNR in the most
			# sensitive instrument) at which we will add weight
			# to the PDF.  from the SNR in fastest growing
			# instrument, the distance to the source is:
			#
			#	D = max_snr_times_D / snr
			#
			# and the SNRs in all instruments are:
			#
			#	snr_times_D / D
			#
			# number of sources:
			#
			#	d count \propto D^2 |dD|
			#	count \propto Dhi^3 - Dlo**3
			for D, Dhi, Dlo in max_snr_times_D / snr_snrlo_snrhi_sequence[start_index:end_index]:
				pdf[tuple(snr_times_D / D)] += Dhi**3. - Dlo**3.

			if progressbar is not None:
				progressbar.increment()
		# check for divide-by-zeros that weren't caught
		assert numpy.isfinite(pdf.array).all()

		# convolve the PDF bin values with a Gaussian kernel whose
		# width in bins is equivalent to \sqrt{2} in SNR at SNR=6.
		# at SNRs where we are interested in marginal detections,
		# the binning is approximately uniform in log(SNR) so a
		# kernel that is Gaussian in bins is close to being
		# \chi^2-distributed with 2 DOF in SNR and so this
		# approximates the effect of noise-induced fluctuations on
		# the SNRs of signals.  at high SNR the window (whose size
		# is fixed in bin counts) becomes enormous but at high SNR
		# we need a larger physical kernel to fill in the lower
		# density of samples that have falled out there.  thus this
		# kernel is serving a dual role:  at low SNR it simulates
		# Gaussian noise fluctuations in recovered SNRs and at high
		# SNR it plays the role of a density estimation kernel.
		bins_per_snr_at_6 = [1. / (upper[i] - lower[i]) for i, upper, lower in zip(pdf.bins[(6.,) * len(instruments)], pdf.bins.upper(), pdf.bins.lower())]
		rate.filter_array(pdf.array, rate.gaussian_window(*(math.sqrt(2.) * x for x in bins_per_snr_at_6)))
		# protect against round-off in FFT convolution leading to
		# negative values in the PDF
		numpy.clip(pdf.array, 0., PosInf, pdf.array)
		# set the region where any SNR is lower than the input
		# threshold to zero before normalizing the pdf and
		# returning.
		range_all = slice(None, None)
		range_low = slice(None, cls.snr_min)
		for i in xrange(len(instruments)):
			slices = [range_all] * len(instruments)
			slices[i] = range_low
			pdf[tuple(slices)] = 0.
		pdf.to_pdf()
		return pdf


def P_instruments_given_signal(horizon_history, n_samples = 500000, min_distance = 0.):
	# FIXME:  this function computes P(instruments | signal)
	# marginalized over time (i.e., marginalized over the history of
	# horizon distances).  what we really want is to know P(instruments
	# | signal, horizon distances), that is we want to leave it
	# depending parametrically on the instantaneous horizon distances.
	# this function takes about 30 s to evaluate, so computing it on
	# the fly isn't practical and we'll require some sort of caching
	# scheme.  unless somebody can figure out how to compute this
	# explicitly without resorting to Monte Carlo integration.

	# FIXME:  this function does not yet incorporate the effect of
	# noise-induced SNR fluctuations in its calculations

	if n_samples < 1:
		raise ValueError("n_samples=%d must be >= 1" % n_samples)
	if min_distance < 0.:
		raise ValueError("min_distance=%g must be >= 0" % min_distance)

	# get instrument names
	names = tuple(horizon_history)
	if not names:
		raise ValueError("horizon_history is empty")
	# get responses in that same order
	resps = [inject.cached_detector[inject.prefix_to_name[inst]].response for inst in names]

	# initialize output.  dictionary mapping instrument combination to
	# probability (initially all 0).
	result = dict.fromkeys((frozenset(instruments) for n in xrange(2, len(names) + 1) for instruments in iterutils.choices(names, n)), 0.0)

	# psi chooses the orientation of F+ and Fx, choosing a fixed value
	# is OK.  we select random uniformly-distributed right ascensions
	# so there's no point in also choosing random GMSTs and any value
	# is as good as any other
	psi = gmst = 0.0

	# function to spit out a choice of horizon distances drawn
	# uniformly in time
	rand_horizon_distances = iter(horizon_history.randhorizons()).next

	# in the loop, we'll need a sequence of integers to enumerate
	# instruments.  construct it here to avoid doing it repeatedly in
	# the loop
	indexes = tuple(range(len(names)))

	# avoid attribute look-ups and arithmetic in the loop
	acos = math.acos
	numpy_array = numpy.array
	numpy_dot = numpy.dot
	numpy_sqrt = numpy.sqrt
	random_uniform = random.uniform
	twopi = 2. * math.pi
	pi_2 = math.pi / 2.
	xlal_am_resp = inject.XLALComputeDetAMResponse

	# loop as many times as requested
	for i in xrange(n_samples):
		# retrieve random horizon distances in the same order as
		# the instruments.  note:  rand_horizon_distances() is only
		# evaluated once in this expression.  that's important
		DH = numpy_array(map(rand_horizon_distances().__getitem__, names))

		# select random sky location and source orbital plane
		# inclination
		theta = acos(random_uniform(-1., 1.))
		phi = random_uniform(0., twopi)
		cosi2 = random_uniform(-1., 1.)**2.

		# compute F+^2 and Fx^2 for each antenna from the sky
		# location and antenna responses
		fpfc2 = numpy_array(tuple(xlal_am_resp(resp, phi, pi_2 - theta, psi, gmst) for resp in resps))**2.

		# 1/8 ratio of inverse SNR to distance for each instrument
		# (1/8 because horizon distance is defined for an SNR of 8,
		# and we've omitted that factor for performance)
		snr_times_D_over_8 = DH * numpy_sqrt(numpy_dot(fpfc2, ((1. + cosi2)**2. / 4., cosi2)))

		# the volume visible to each instrument given the
		# requirement that a source be above the SNR threshold is
		#
		# V = [constant] * (8 * snr_times_D_over_8 / snr_threshold)**3.
		#
		# but in the end we'll only need ratios of these volumes so
		# we can omit the proportionality constant and we can also
		# omit the factor of (8 / snr_threshold)**3
		V_at_snr_threshold = snr_times_D_over_8**3.

		# order[0] is index of instrument that can see sources the
		# farthest, order[1] is index of instrument that can see
		# sources the next farthest, etc.
		order = sorted(indexes, key = V_at_snr_threshold.__getitem__, reverse = True)
		ordered_names = tuple(names[i] for i in order)

		# instrument combination and volume of space (up to
		# irrelevant proportionality constant) visible to that
		# combination given the requirement that a source be above
		# the SNR threshold in that combination.  sequence of
		# instrument combinations is left as a generator expression
		# for lazy evaluation
		instruments = (frozenset(ordered_names[:n]) for n in xrange(2, len(order) + 1))
		V = tuple(V_at_snr_threshold[i] for i in order[1:])
		if V[0] <= min_distance:
			# fewer than two instruments are on, so no
			# combination can see anything
			continue

		# for each instrument combination, probability that a
		# source visible to at least two instruments is visible to
		# that combination (here is where the proportionality
		# constant and factor of (8/snr_threshold)**3 drop out of
		# the calculation)
		P = tuple(x / V[0] for x in V)

		# accumulate result.  p - pnext is the probability that a
		# source (that is visible to at least two instruments) is
		# visible to that combination of instruments and not any
		# other combination of instruments.
		for key, p, pnext in zip(instruments, P, P[1:] + (0.,)):
			result[key] += p - pnext
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


def binned_likelihood_ratio_rates_from_samples(signal_rates, noise_rates, samples, nsamples):
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
	sample_func = iter(samples).next
	for i in xrange(nsamples):
		lamb, lnP_signal, lnP_noise = sample_func()
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
	# likelihood ratio binning
	#

	binnings = {
		"likelihood_ratio": rate.NDBins((rate.ATanLogarithmicBins(math.exp(0.), math.exp(80.), 5000),))
	}

	filters = {
		"likelihood_ratio": rate.gaussian_window(8.)
	}

	#
	# Threshold at which FAP & FAR normalization will occur
	#

	likelihood_ratio_threshold = math.exp(5.)


	def __init__(self, coinc_params_distributions, instruments = None, process_id = None, nsamples = 1000000, verbose = False):
		self.background_likelihood_rates = {}
		self.background_likelihood_pdfs = {}
		self.signal_likelihood_rates = {}
		self.signal_likelihood_pdfs = {}
		self.zero_lag_likelihood_rates = {}
		self.zero_lag_likelihood_pdfs = {}
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

		# initialize binnings
		for key in [frozenset(ifos) for n in range(2, len(instruments) + 1) for ifos in iterutils.choices(instruments, n)]:
			self.background_likelihood_rates[key] = rate.BinnedArray(self.binnings["likelihood_ratio"])
			self.signal_likelihood_rates[key] = rate.BinnedArray(self.binnings["likelihood_ratio"])
			self.zero_lag_likelihood_rates[key] = rate.BinnedArray(self.binnings["likelihood_ratio"])

		# calculate all of the possible ifo combinations with at least
		# 2 detectors in order to get the joint likelihood pdfs
		likelihoodratio_func = snglcoinc.LikelihoodRatio(coinc_params_distributions)
		threads = []
		for key in self.background_likelihood_rates:
			if verbose:
				print >>sys.stderr, "computing signal and noise likelihood PDFs for %s" % ", ".join(sorted(key))
			q = multiprocessing.queues.SimpleQueue()
			p = multiprocessing.Process(target = lambda: q.put(binned_likelihood_ratio_rates_from_samples(self.signal_likelihood_rates[key], self.background_likelihood_rates[key], self.likelihoodratio_samples(iter(coinc_params_distributions.random_params(key)).next, likelihoodratio_func, coinc_params_distributions.lnP_signal, coinc_params_distributions.lnP_noise), nsamples = nsamples)))
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


	def collect_zero_lag_rates(self, connection, coinc_def_id):
		for instruments, likelihood_ratio in connection.cursor().execute("""
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
			assert likelihood_ratio is not None, "null likelihood ratio encountered.  probably coincs have not been ranked"
			self.zero_lag_likelihood_rates[frozenset(lsctables.instrument_set_from_ifos(instruments))][likelihood_ratio,] += 1.

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


	@staticmethod
	def likelihoodratio_samples(random_params_func, likelihoodratio_func, lnP_signal_func, lnP_noise_func):
		"""
		Generator that yields an unending sequence of 3-element
		tuples.  Each tuple's elements are a value of the
		likelihood rato, the natural log of the probability density
		of that likelihood ratio in the signal population, the
		natural log of the probability density of that likelihood
		ratio in the noise population.
		"""
		while 1:
			params, lnP_params = random_params_func()
			lamb = likelihoodratio_func(params)
			if math.isnan(lamb):
				raise ValueError("encountered NaN likelihood ratio at %s" % repr(params))
			yield lamb, lnP_signal_func(params) - lnP_params, lnP_noise_func(params) - lnP_params

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
		def build_pdf(binnedarray, likelihood_ratio_threshold, filt):
			# copy counts into pdf array and smooth
			pdf = binnedarray.copy()
			rate.filter_array(pdf.array, filt)
			# zero the counts below the threshold.  need to
			# make sure the bin @ threshold is also 0'ed
			pdf[:likelihood_ratio_threshold,] = 0.
			pdf[likelihood_ratio_threshold,] = 0.
			# zero the counts in the infinite-sized high bin so
			# the final PDF normalization ends up OK
			pdf.array[-1] = 0.
			# convert to normalized PDF
			pdf.to_pdf()
			return pdf
		if verbose:
			progressbar = ProgressBar(text = "Computing Lambda PDFs", max = len(self.background_likelihood_rates) + len(self.signal_likelihood_rates) + len(self.zero_lag_likelihood_rates))
			progressbar.show()
		else:
			progressbar = None
		for key, binnedarray in self.background_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s noise model likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.background_likelihood_pdfs[key] = build_pdf(binnedarray, self.likelihood_ratio_threshold, self.filters["likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()
		for key, binnedarray in self.signal_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s signal model likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.signal_likelihood_pdfs[key] = build_pdf(binnedarray, self.likelihood_ratio_threshold, self.filters["likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()
		for key, binnedarray in self.zero_lag_likelihood_rates.items():
			assert not numpy.isnan(binnedarray.array).any(), "%s zero lag likelihood ratio counts contain NaNs" % (key if key is not None else "combined")
			self.zero_lag_likelihood_pdfs[key] = build_pdf(binnedarray, self.likelihood_ratio_threshold, self.filters["likelihood_ratio"])
			if progressbar is not None:
				progressbar.increment()

	def __iadd__(self, other):
		snglcoinc.CoincParamsDistributions.addbinnedarrays(self.background_likelihood_rates, other.background_likelihood_rates, self.background_likelihood_pdfs, other.background_likelihood_pdfs)
		snglcoinc.CoincParamsDistributions.addbinnedarrays(self.signal_likelihood_rates, other.signal_likelihood_rates, self.signal_likelihood_pdfs, other.signal_likelihood_pdfs)
		snglcoinc.CoincParamsDistributions.addbinnedarrays(self.zero_lag_likelihood_rates, other.zero_lag_likelihood_rates, self.zero_lag_likelihood_pdfs, other.zero_lag_likelihood_pdfs)
		return self

	@classmethod
	def from_xml(cls, xml, name):
		# find the root of the XML tree containing the
		# serialization of this object
		xml, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == u"%s:%s" % (name, cls.ligo_lw_name_suffix)]

		# create a mostly uninitialized instance
		self = cls(None, {}, process_id = ligolw_param.get_pyvalue(xml, u"process_id"))

		# pull out the likelihood count and PDF arrays
		def reconstruct(xml, prefix, target_dict):
			for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and ("_%s" % prefix) in elem.Name]:
				ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.Name.split("_")[0]))
				target_dict[ifo_set] = rate.binned_array_from_xml(ba_elem, ba_elem.Name.split(":")[0])
		reconstruct(xml, u"background_likelihood_rate", self.background_likelihood_rates)
		reconstruct(xml, u"background_likelihood_pdf", self.background_likelihood_pdfs)
		reconstruct(xml, u"signal_likelihood_rate", self.signal_likelihood_rates)
		reconstruct(xml, u"signal_likelihood_pdf", self.signal_likelihood_pdfs)
		reconstruct(xml, u"zero_lag_likelihood_rate", self.zero_lag_likelihood_rates)
		reconstruct(xml, u"zero_lag_likelihood_pdf", self.zero_lag_likelihood_pdfs)

		assert set(self.background_likelihood_rates) == set(self.background_likelihood_pdfs)
		assert set(self.signal_likelihood_rates) == set(self.signal_likelihood_pdfs)
		assert set(self.zero_lag_likelihood_rates) == set(self.zero_lag_likelihood_pdfs)
		assert set(self.background_likelihood_rates) == set(self.signal_likelihood_rates)
		assert set(self.background_likelihood_rates) == set(self.zero_lag_likelihood_rates)

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
		store(xml, u"zero_lag_likelihood_rate", self.zero_lag_likelihood_rates)
		store(xml, u"zero_lag_likelihood_pdf", self.zero_lag_likelihood_pdfs)

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
		# implements equation (8) from Phys. Rev. D 88, 024025.
		# arXiv:1209.0718
		rank = max(self.minrank, min(self.maxrank, rank))
		fap = float(self.ccdf_interpolator[None](rank))
		return fap_after_trials(fap, self.count_above_threshold[None])

	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.  the return value is divided by T to
		# convert events/experiment to events/second.
		assert self.livetime is not None, "cannot compute FAR without livetime"
		rank = max(self.minrank, min(self.maxrank, rank))
		# true-dismissal probability = 1 - single-event false-alarm
		# probability, the integral in equation (B4)
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
	from scipy.optimize import fmin
	# the upper bound is chosen to include N + \sqrt{N}
	return fmin((lambda x: -RatesLnPDF(x, f_over_b)), (1.0, len(f_over_b) + len(f_over_b)**.5), disp = True)


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

	pos0 is an n_walker by n_dim array giving the initial positions of
	the walkers (this parameter is currently not optional).  n_burn
	iterations of the MCMC sampler will be executed and discarded to
	allow the system to stabilize before samples are yielded to the
	calling code.

	NOTE:  the samples yielded by this generator are not drawn from the
	PDF independently of one another.  The correlation length is not
	known at this time.
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
	nbins = len(samples) // 729
	binnedarray = rate.BinnedArray(rate.NDBins((rate.LinearBins(lo, hi, nbins),)))
	for sample in samples:
		binnedarray[sample,] += 1.
	rate.filter_array(binnedarray.array, rate.gaussian_window(5))
	numpy.clip(binnedarray.array, 0.0, PosInf, binnedarray.array)
	return binnedarray


def calculate_rate_posteriors(ranking_data, likelihood_ratios, restrict_to_instruments = None, progressbar = None):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if any(math.isnan(lr) for lr in likelihood_ratios):
		raise ValueError("NaN likelihood ratio encountered")
	# FIXME;  can't we handle this next case somehow?
	if any(math.isinf(lr) for lr in likelihood_ratios):
		raise ValueError("infinite likelihood ratio encountered")
	if any(lr < 0. for lr in likelihood_ratios):
		raise ValueError("negative likeklihood ratio encountered")

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
	f_over_b = numpy.array([ranking_data.signal_likelihood_pdfs[restrict_to_instruments][likelihood_ratios[index],] / ranking_data.background_likelihood_pdfs[restrict_to_instruments][likelihood_ratios[index],] for index in order])

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
	nsample = 400000
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
		if progressbar is not None:
			progressbar.update(progressbar.max)
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
