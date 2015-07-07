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
# | Names                                          | Hash                                        | Date       | Diff to Head of Master      |
# | -------------------------------------------    | ------------------------------------------- | ---------- | --------------------------- |
# | Hanna, Cannon, Meacher, Creighton J, Robinet, Sathyaprakash, Messick, Dent, Blackburn | 7fb5f008afa337a33a72e182d455fdd74aa7aa7a | 2014-11-05 |<a href="@gstlal_inspiral_cgit_diff/python/far.py?id=HEAD&id2=7fb5f008afa337a33a72e182d455fdd74aa7aa7a">far.py</a> |
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
# FIXME remove this when the LDG upgrades scipy on the SL6 systems, Debian
# systems are already fine
try:
	from scipy.optimize import curve_fit
except ImportError:
	from gstlal.curve_fit import curve_fit
from scipy import stats
from scipy.special import ive
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
import lal
import lalsimulation
from pylal import rate
from pylal import snglcoinc


#
# ============================================================================
#
#			Non-central chisquared pdf
#
# ============================================================================
#

#
# FIXME this is to work around a precision issue in scipy
# See: https://github.com/scipy/scipy/issues/1608
#

def logiv(v, z):
	# from Abramowitz and Stegun (9.7.1), if mu = 4 v^2, then for large
	# z:
	#
	# Iv(z) ~= exp(z) / (\sqrt(2 pi z)) { 1 - (mu - 1)/(8 z) + (mu - 1)(mu - 9) / (2! (8 z)^2) - (mu - 1)(mu - 9)(mu - 25) / (3! (8 z)^3) ... }
	# Iv(z) ~= exp(z) / (\sqrt(2 pi z)) { 1 + (mu - 1)/(8 z) [-1 + (mu - 9) / (2 (8 z)) [1 - (mu - 25) / (3 (8 z)) ... ]]}
	# log Iv(z) ~= z - .5 log(2 pi) log z + log1p((mu - 1)/(8 z) (-1 + (mu - 9)/(16 z) (1 - (mu - 25)/(24 z) ... )))

	with numpy.errstate(divide = "ignore"):
		a = numpy.log(ive(v,z))

	# because this result will only be used for large z, to silence
	# divide-by-0 complaints from inside log1p() when z is small we
	# clip z to 1.
	mu = 4. * v**2.
	with numpy.errstate(divide = "ignore", invalid = "ignore"):
		b = -math.log(2. * math.pi) / 2. * numpy.log(z)
		z = numpy.clip(z, 1., PosInf)
		b += numpy.log1p((mu - 1.) / (8. * z) * (-1. + (mu - 9.) / (16. * z) * (1. - (mu - 25.) / (24. * z))))

	return z + numpy.where(z < 1e8, a, b)

# See: http://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
def ncx2logpdf(x, k, l):
	return -math.log(2.) - (x+l)/2. + (k/4.-.5) * (numpy.log(x) - numpy.log(l)) + logiv(k/2.-1., numpy.sqrt(l) * numpy.sqrt(x))

def ncx2pdf(x, k, l):
	return numpy.exp(ncx2logpdf(x, k, l))


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

	Example:

	>>> fap_after_trials(0.5, 1)
	0.5
	>>> fap_after_trials(0.066967008463192584, 10)
	0.5
	>>> fap_after_trials(0.0069075045629640984, 100)
	0.5
	>>> fap_after_trials(0.00069290700954747807, 1000)
	0.5
	>>> fap_after_trials(0.000069312315846428086, 10000)
	0.5
	>>> fap_after_trials(.000000006931471781576803, 100000000)
	0.5
	>>> fap_after_trials(0.000000000000000069314718055994534, 10000000000000000)
	0.5
	>>> "%.15g" % fap_after_trials(0.1, 21.854345326782834)
	'0.9'
	>>> "%.15g" % fap_after_trials(1e-17, 2.3025850929940458e+17)
	'0.9'
	>>> fap_after_trials(0.1, .2)
	0.020851637639023216
	"""
	# when m*p is not >> 1, we can use an expansion in powers of m*p
	# that allows us to avoid having to compute differences of
	# probabilities (which would lead to loss of precision when working
	# with probabilities close to 0 or 1)
	#
	# 1 - (1 - p)^m = m p - (m^2 - m) p^2 / 2 +
	#	(m^3 - 3 m^2 + 2 m) p^3 / 6 -
	#	(m^4 - 6 m^3 + 11 m^2 - 6 m) p^4 / 24 + ...
	#
	# the coefficient of each power of p is a polynomial in m.  if m <<
	# 1, these polynomials can be approximated by their leading term in
	# m
	#
	# 1 - (1 - p)^m ~= m p + m p^2 / 2 + m p^3 / 3 + m p^4 / 4 + ...
	#                = -m * log(1 - p)
	#
	# NOTE: the ratio of the coefficients of higher-order terms in the
	# polynomials in m to that of the leading-order term grows with
	# successive powers of p and eventually the higher-order terms will
	# dominate.  we assume that this does not happen until the power of
	# p is sufficiently high that the entire term (polynomial in m and
	# all) is negligable and the series has been terminated.
	#
	# if m is not << 1, then returning to the full expansion, starting
	# at 0, the nth term in the series is
	#
	# -1^n * (m - 0) * (m - 1) * ... * (m - n) * p^(n + 1) / (n + 1)!
	#
	# if the (n-1)th term is X, the nth term in the series is
	#
	# X * (n - m) * p / (n + 1)
	#
	# this recursion relation allows us to compute successive terms
	# without explicit evaluation of the full numerator and denominator
	# expressions (each of which quickly overflow).
	#
	# for sufficiently large n the denominator dominates and the terms
	# in the series become small and the sum eventually converges to a
	# stable value (and if m is an integer the sum is exact in a finite
	# number of terms).  however, if m*p >> 1 it can take many terms
	# before the series sum stabilizes, terms in the series initially
	# grow large and alternate in sign and an accurate result can only
	# be obtained through careful cancellation of the large values.
	#
	# for large m*p we write the expression as
	#
	# 1 - (1 - p)^m = 1 - exp(m log(1 - p))
	#
	# if p is small, log(1 - p) suffers from loss of precision but the
	# Taylor expansion of log(1 - p) converges quickly
	#
	# m ln(1 - p) = -m p - m p^2 / 2 - m p^3 / 3 - ...
	#             = -m p * (1 + p / 2 + p^2 / 3 + ...)
	#
	# math libraries (including Python's) generally provide log1p(),
	# which evalutes log(1 + p) accurately for small p.  we rely on
	# this function to provide results valid both in the small-p and
	# not small-p regimes.
	#
	# if p = 1, log1p() complains about an overflow error.  we trap
	# these and return the hard-coded answer
	#

	if m <= 0.:
		raise ValueError("m = %g must be positive" % m)
	if not (0. <= p <= 1.):
		raise ValueError("p = %g must be between 0 and 1 inclusively" % p)

	if m * p < 4.:
		#
		# expansion of 1 - (1 - p)^m in powers of p
		#

		if m < 1e-8:
			#
			# small m approximation
			#

			try:
				return -m * math.log1p(-p)
			except OverflowError:
				#
				# p is too close to 1.  result is 1
				#

				return 1.

		#
		# general m
		#

		s = []
		term = -1.
		for n in itertools.count():
			term *= (n - m) * p / (n + 1.)
			s.append(term)
			# 0th term is always positive
			if abs(term) <= 1e-18 * s[0]:
				s.reverse()
				return sum(s)

	#
	# compute result as 1 - exp(m * log(1 - p))
	#

	try:
		x = m * math.log1p(-p)
	except OverflowError:
		#
		# p is very close to 1.  we know p <= 1 because it's a
		# probability, and we know that m*p >= 4 otherwise we
		# wouldn't have followed this code path, and so because p
		# is close to 1 and m is not small we can safely assume the
		# anwer is 1.
		#

		return 1.

	if x > -0.69314718055994529:
		#
		# result is closer to 0 than to 1.  use Taylor expansion
		# for exp() with leading term removed to avoid having to
		# subtract from 1 to get answer.
		#
		# 1 - exp x = -(x + x^2/2! + x^3/3! + ...)
		#

		s = [x]
		term = x
		for n in itertools.count(2):
			term *= x / n
			s.append(term)
			# 0th term is always negative
			if abs(term) <= -1e-18 * s[0]:
				s.reverse()
				return -sum(s)

	return 1. - math.exp(x)


fap_after_trials_arr = numpy.frompyfunc(fap_after_trials, 2, 1)


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
	# m log(1 - p0) = log(1 - p1)
	#

	return math.log1p(-p1) / math.log1p(-p0)


#
# =============================================================================
#
#                 Parameter Distributions Book-Keeping Object
#
# =============================================================================
#


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
		return val_lo if abs(x - x_lo) < abs(x_hi - x) else val_hi

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


class CoincParams(dict):
	# place-holder class to allow params dictionaries to carry
	# attributes as well
	__slots__ = ("horizons",)


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
	# if the smaller of the two is < min_ratio * the larger then the
	# smaller is treated as though it were 0.
	# FIXME:  is this choice of distance quantization appropriate?
	@staticmethod
	def quantize_horizon_distances(horizon_distances, log_distance_tolerance = PosInf, min_ratio = 0.1):
		horizon_distance_norm = max(horizon_distances.values())
		assert horizon_distance_norm != 0.
		if math.isinf(log_distance_tolerance):
			return dict((instrument, 1.) for instrument in horizon_distances)
		min_distance = min_ratio * horizon_distance_norm
		return dict((instrument, (0. if horizon_distance < min_distance else math.exp(round(math.log(horizon_distance / horizon_distance_norm) / log_distance_tolerance) * log_distance_tolerance))) for instrument, horizon_distance in horizon_distances.items())

	# binnings (filter funcs look-up initialized in .__init__()
	snr_chi_binning = rate.NDBins((rate.ATanLogarithmicBins(3.6, 70., 600), rate.ATanLogarithmicBins(.001, 0.5, 300)))
	binnings = {
		"instruments": rate.NDBins((rate.LinearBins(0.5, instrument_categories.max() + 0.5, instrument_categories.max()),)),
		"H1_snr_chi": snr_chi_binning,
		"H2_snr_chi": snr_chi_binning,
		"H1H2_snr_chi": snr_chi_binning,
		"L1_snr_chi": snr_chi_binning,
		"V1_snr_chi": snr_chi_binning,
		"E1_snr_chi": snr_chi_binning,
		"E2_snr_chi": snr_chi_binning,
		"E3_snr_chi": snr_chi_binning,
		"E0_snr_chi": snr_chi_binning
	}
	del snr_chi_binning

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
			"E1_snr_chi": self.pdf_from_rates_snrchi2,
			"E2_snr_chi": self.pdf_from_rates_snrchi2,
			"E3_snr_chi": self.pdf_from_rates_snrchi2,
			"E0_snr_chi": self.pdf_from_rates_snrchi2,
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
	#		frozenset(("H1", ...)), frozenset([("H1", horiz_dist), ("L1", horiz_dist), ...]): (InterpBinnedArray, BinnedArray, age),
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
		# largest among them and then the fractions aquantized to
		# integer powers of a common factor
		#

		key = frozenset(instruments), frozenset(self.quantize_horizon_distances(horizon_distances).items())

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
			del progressbar
			lnbinnedarray = binnedarray.copy()
			with numpy.errstate(divide = "ignore"):
				lnbinnedarray.array = numpy.log(lnbinnedarray.array)
			pdf = rate.InterpBinnedArray(lnbinnedarray, fill_value = NegInf)
			self.snr_joint_pdf_cache[key] = pdf, binnedarray, age
			# if the cache is full, delete the entry with the
			# smallest age
			while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
				del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
			if verbose:
				print >>sys.stderr, "%d/%d slots in SNR PDF cache now in use" % (len(self.snr_joint_pdf_cache), self.max_cached_snr_joint_pdfs)
		return pdf

	def coinc_params(self, events, offsetvector):
		#
		# NOTE:  unlike the burst codes, this function is expected
		# to work with single-instrument event lists as well, as
		# it's output is used to populate the single-instrument
		# background bin counts.
		#

		#
		# 2D (snr, \chi^2) values.  don't allow both H1 and H2 to
		# both contribute parameters to the same coinc.  if both
		# have participated favour H1
		#

		params = CoincParams(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events)
		if "H2_snr_chi" in params and "H1_snr_chi" in params:
			del params["H2_snr_chi"]

		#
		# instrument combination
		#

		params["instruments"] = (ThincaCoincParamsDistributions.instrument_categories.category(event.ifo for event in events),)

		#
		# record the horizon distances.  pick one trigger at random
		# to provide a timestamp and pull the horizon distances
		# from our horizon distance history at that time.  the
		# horizon history is keyed by floating-point values (don't
		# need nanosecond precision for this).  NOTE:  this is
		# attached as a property instead of going into the
		# dictionary to not confuse the stock lnP_noise(),
		# lnP_signal(), and friends methods.
		#

		params.horizons = self.horizon_history.getdict(float(events[0].get_end()))
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

	def lnP_signal(self, params):
		# (instrument, snr) pairs sorted alphabetically by
		# instrument name
		snrs = sorted((name.split("_")[0], value[0]) for name, value in params.items() if name.endswith("_snr_chi"))
		# retrieve the SNR PDF
		snr_pdf = self.get_snr_joint_pdf((instrument for instrument, rho in snrs), params.horizons)
		# evaluate it (snrs are alphabetical by instrument)
		lnP_signal = snr_pdf(*tuple(rho for instrument, rho in snrs))

		# FIXME:  P(instruments | signal) needs to depend on
		# horizon distances.  here we're assuming whatever
		# populate_prob_of_instruments_given_signal() has set the
		# probabilities to is OK.  we probably need to cache these
		# and save them in the XML file, too, like P(snrs | signal,
		# instruments)
		lnP_signal += super(ThincaCoincParamsDistributions, self).lnP_signal(params)

		# return logarithm of (.99 P(..|signal) + 0.01 P(..|noise))
		# FIXME:  investigate how to determine correct mixing ratio
		lnP_noise = self.lnP_noise(params)
		if math.isinf(lnP_noise) and math.isinf(lnP_signal):
			if lnP_noise < 0. and lnP_signal < 0.:
				return NegInf
			if lnP_noise > 0. and lnP_signal > 0.:
				return PosInf
		lnP_signal += math.log(.99)
		lnP_noise += math.log(0.01)
		return max(lnP_signal, lnP_noise) + math.log1p(math.exp(-abs(lnP_signal - lnP_noise)))

	def add_snrchi_prior(self, rates_dict, n, prefactors_range, df, inv_snr_pow = 4., verbose = False):
		if verbose:
			print >>sys.stderr, "synthesizing signal-like (SNR, \\chi^2) distributions ..."
		if df <= 0.:
			raise ValueError("require df >= 0: %s" % repr(df))
		pfs = numpy.logspace(numpy.log10(prefactors_range[0]), numpy.log10(prefactors_range[1]), 100)
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
				new_binarr.array[snrindices, rcossindices] += ncx2pdf(snrchi2, df, numpy.array([pf * snrs2]).T)

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

		if verbose:
			print >>sys.stderr, "adding tilt to (SNR, \\chi^2) background PDFs ..."
		for instrument, number_of_events in n.items():
			binarr = getattr(self, ba)["%s_snr_chi" % instrument]

			# will need to normalize results so need new
			# storage
			new_binarr = rate.BinnedArray(binarr.bins)

			snr = new_binarr.bins[0].centres()
			rcoss = new_binarr.bins[1].centres()

			# ignore overflows in SNR^6.  correct answer is 0
			# when that happens.
			with numpy.errstate(over = "ignore"):
				psnr = numpy.exp(-(snr**2 - 6.**2)/ 2.) + (1 + 6.**6) / (1. + snr**6)
			prcoss = numpy.ones(len(rcoss))

			# the bins at the edges end up with infinite volume
			# elements.  the PDF should be 0 in those bins so
			# we 0 their volume elements to force that result
			dsnr_drcoss = new_binarr.bins.volumes()
			dsnr_drcoss[~numpy.isfinite(dsnr_drcoss)] = 0.
			new_binarr.array[:,:] = numpy.outer(psnr, prcoss) * dsnr_drcoss

			# Normalize what's left to the requested count.
			# Give .1% of the requested events to this portion of the model
			new_binarr.array *= 0.001 * number_of_events / new_binarr.array.sum()
			# add to raw counts
			getattr(self, ba)["instruments"][self.instrument_categories.category(frozenset([instrument])),] += number_of_events
			binarr += new_binarr

		# Give 99.9% of the requested events to the "glitch model"
		self.add_snrchi_prior(getattr(self, ba), dict((ifo, x * 0.999) for ifo, x in n.items()), prefactors_range = prefactors_range, df = df, inv_snr_pow = inv_snr_pow, verbose = verbose)

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
		assert N > 0
		assert coincidence_bins > 0.
		coincidence_bins /= N
		if verbose:
			print >>sys.stderr, "\tthere seems to be %g effective disjoint coincidence bin(s)" % coincidence_bins
		assert coincidence_bins >= 1.
		# convert single-instrument event rates to rates/bin
		coincsynth.mu = dict((instrument, rate / coincidence_bins) for instrument, rate in coincsynth.mu.items())
		# now compute the expected coincidence rates/bin, then
		# multiply by the number of bins to get the expected
		# coincidence rates
		for instruments, count in coincsynth.mean_coinc_count.items():
			self.background_rates["instruments"][self.instrument_categories.category(instruments),] = count * coincidence_bins

	def add_foreground_snrchi_prior(self, n, prefactors_range = (0.01, 0.25), df = 40, inv_snr_pow = 4., verbose = False):
		for instrument, number_of_events in n.items():
			# NOTE a uniform prior is added that must be smaller than the uniform prior added for the background
			self.injection_rates["%s_snr_chi" % instrument].array += 1. / number_of_events / self.injection_rates["%s_snr_chi" % instrument].array.size / 1e20
		self.add_snrchi_prior(self.injection_rates, n, prefactors_range, df, inv_snr_pow = inv_snr_pow, verbose = verbose)

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
		# the instrument combination "interpolators" are
		# pass-throughs.  we pre-evaluate a bunch of attribute,
		# dictionary, and method look-ups for speed
		#

		def mkinterp(binnedarray):
			with numpy.errstate(divide = "ignore"):
				# need to insert an element at the start to
				# get the binning indexes to map the
				# correct locations in the array
				return numpy.hstack(([NaN], numpy.log(binnedarray.array))).__getitem__
		if "instruments" in self.background_pdf:
			self.background_lnpdf_interp["instruments"] = mkinterp(self.background_pdf["instruments"])
		if "instruments" in self.injection_pdf:
			self.injection_lnpdf_interp["instruments"] = mkinterp(self.injection_pdf["instruments"])
		if "instruments" in self.zero_lag_pdf:
			self.zero_lag_lnpdf_interp["instruments"] = mkinterp(self.zero_lag_pdf["instruments"])

	def pdf_from_rates_instruments(self, key, pdf_dict):
		# instrument combos are probabilities, not densities.  be
		# sure the single-instrument categories are zeroed.
		binnedarray = pdf_dict[key]
		for category in self.instrument_categories.values():
			binnedarray[category,] = 0
		with numpy.errstate(invalid = "ignore"):
			binnedarray.array /= binnedarray.array.sum()

	def pdf_from_rates_snrchi2(self, key, pdf_dict, snr_kernel_width_at_8 = 10., chisq_kernel_width = 0.1,  sigma = 10.):
		# get the binned array we're going to process
		binnedarray = pdf_dict[key]
		numsamples = binnedarray.array.sum() / 10. + 1. # Be extremely conservative and assume only 1 in 10 samples are independent.
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

		# Compute the KDE
		kernel = rate.gaussian_window(snr_kernel_bins, chisq_kernel_bins, sigma = sigma)

		# convolve with the bin count data
		rate.filter_array(binnedarray.array, kernel)

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

	@classmethod
	def from_xml(cls, xml, name):
		self = super(ThincaCoincParamsDistributions, cls).from_xml(xml, name)
		xml = self.get_xml_root(xml, name)
		self.horizon_history = HorizonHistories.from_xml(xml, name)
		prefix = u"cached_snr_joint_pdf"
		for elem in [elem for elem in xml.childNodes if elem.Name.startswith(u"%s:" % prefix)]:
			key = ligolw_param.get_pyvalue(elem, u"key").strip().split(u";")
			key = frozenset(lsctables.instrument_set_from_ifos(key[0].strip())), frozenset((inst.strip(), float(dist.strip())) for inst, dist in (inst_dist.strip().split(u"=") for inst_dist in key[1].strip().split(u",")))
			binnedarray = rate.BinnedArray.from_xml(elem, prefix)
			if self.snr_joint_pdf_cache:
				age = max(age for ignored, ignored, age in self.snr_joint_pdf_cache.values()) + 1
			else:
				age = 0
			lnbinnedarray = binnedarray.copy()
			with numpy.errstate(divide = "ignore"):
				lnbinnedarray.array = numpy.log(lnbinnedarray.array)
			self.snr_joint_pdf_cache[key] = rate.InterpBinnedArray(lnbinnedarray, fill_value = NegInf), binnedarray, age
			while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
				del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
		return self

	def to_xml(self, name):
		xml = super(ThincaCoincParamsDistributions, self).to_xml(name)
		xml.appendChild(self.horizon_history.to_xml(name))
		prefix = u"cached_snr_joint_pdf"
		for key, (ignored, binnedarray, ignored) in self.snr_joint_pdf_cache.items():
			elem = xml.appendChild(binnedarray.to_xml(prefix))
			elem.appendChild(ligolw_param.new_param(u"key", u"lstring", "%s;%s" % (lsctables.ifos_from_instrument_set(key[0]), u",".join(u"%s=%.17g" % inst_dist for inst_dist in sorted(key[1])))))
		return xml

	@property
	def count_above_threshold(self):
		"""
		Dictionary mapping instrument combination (as a frozenset)
		to number of zero-lag coincs observed.  An additional entry
		with key None stores the total.
		"""
		count_above_threshold = dict((frozenset(self.instrument_categories.instruments(int(round(category)))), count) for category, count in zip(self.zero_lag_rates["instruments"].bins.centres()[0], self.zero_lag_rates["instruments"].array))
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

		See also:

		random_sim_params()

		The sequence is suitable for input to the
		pylal.snglcoinc.LnLikelihoodRatio.samples() log likelihood
		ratio generator.
		"""
		snr_slope = 0.8 / len(instruments)**3

		keys = tuple("%s_snr_chi" % instrument for instrument in instruments)
		base_params = {"instruments": (self.instrument_categories.category(instruments),)}
		horizongen = iter(self.horizon_history.randhorizons()).next
		# P(horizons) = 1/livetime
		log_P_horizons = -math.log(self.horizon_history.maxkey() - self.horizon_history.minkey())
		coordgens = tuple(iter(self.binnings[key].randcoord(ns = (snr_slope, 1.), domain = (slice(self.snr_min, None), slice(None, None)))).next for key in keys)
		while 1:
			seq = sum((coordgen() for coordgen in coordgens), ())
			params = CoincParams(zip(keys, seq[0::2]))
			params.update(base_params)
			params.horizons = horizongen()
			# NOTE:  I think the result of this sum is, in
			# fact, correctly normalized, but nothing requires
			# it to be (only that it be correct up to an
			# unknown constant) and I've not checked that it is
			# so the documentation doesn't promise that it is.
			yield params, sum(seq[1::2], log_P_horizons)

	def random_sim_params(self, sim, horizon_distance = None, snr_min = None, snr_efficiency = 1.0):
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

		The sequence is suitable for input to the
		pylal.snglcoinc.LnLikelihoodRatio.samples() log likelihood
		ratio generator.

		Bugs:

		The second element in each tuple in the sequence is merely
		a placeholder, not the natural logarithm of the PDF from
		which the sample has been drawn, as in the case of
		random_params().  Therefore, when used in combination with
		pylal.snglcoinc.LnLikelihoodRatio.samples(), the two
		probability densities computed and returned by that
		generator along with each log likelihood ratio value will
		simply be the probability densities of the signal and noise
		populations at that point in parameter space.  They cannot
		be used to form an importance weighted sampler of the log
		likeklihood ratios.
		"""
		#
		# retrieve horizon distance from history if not given
		# explicitly.  retrieve SNR threshold from class attribute
		# if not given explicitly
		#

		if horizon_distance is None:
			horizon_distance = self.horizon_history[float(sim.get_time_geocent())]
		if snr_min is None:
			snr_min = self.snr_min

		#
		# compute nominal SNRs
		#
		# FIXME:  remove LIGOTimeGPS type cast when sim is ported
		# to swig bindings
		#

		cosi2 = math.cos(sim.inclination)**2.
		gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(0, sim.get_time_geocent().ns()))
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
			if len(instruments) < 2:
				continue
			params["instruments"] = (ThincaCoincParamsDistributions.instrument_categories.category(instruments),)
			params.horizons = horizon_distance
			yield params, 0.


	@classmethod
	def joint_pdf_of_snrs(cls, instruments, inst_horiz_mapping, n_samples = 160000, bins = rate.ATanLogarithmicBins(3.6, 120., 100), progressbar = None):
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
		resps = tuple(lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in instruments)

		# get horizon distances and responses of remaining
		# instruments (order doesn't matter as long as they're in
		# the same order)
		DH_times_8_other = 8. * numpy.array([dist for inst, dist in inst_horiz_mapping.items() if inst not in instruments])
		resps_other = tuple(lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in inst_horiz_mapping if inst not in instruments)

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

		# we select random uniformly-distributed right ascensions
		# so there's no point in also choosing random GMSTs and any
		# value is as good as any other
		gmst = 0.0

		# run the sampler the requested # of iterations.  save some
		# symbols to avoid doing module attribute look-ups in the
		# loop
		if progressbar is not None:
			progressbar.max = n_samples
		acos = math.acos
		random_uniform = random.uniform
		twopi = 2. * math.pi
		pi_2 = math.pi / 2.
		xlal_am_resp = lal.ComputeDetAMResponse
		# FIXME:  scipy.stats.rice.rvs broken on reference OS.
		# switch to it when we can rely on a new-enough scipy
		#rice_rvs = stats.rice.rvs	# broken on reference OS
		rice_rvs = lambda x: numpy.sqrt(stats.ncx2.rvs(2., x**2.))
		for i in xrange(n_samples):
			# select random sky location and source orbital
			# plane inclination and choice of polarization
			theta = acos(random_uniform(-1., 1.))
			phi = random_uniform(0., twopi)
			psi = random_uniform(0., twopi)
			cosi2 = random_uniform(-1., 1.)**2.

			# F+^2 and Fx^2 for each instrument
			fpfc2 = numpy.array(tuple(xlal_am_resp(resp, phi, pi_2 - theta, psi, gmst) for resp in resps))**2.
			fpfc2_other = numpy.array(tuple(xlal_am_resp(resp, phi, pi_2 - theta, psi, gmst) for resp in resps_other))**2.

			# ratio of distance to inverse SNR for each instrument
			fpfc_factors = ((1. + cosi2)**2. / 4., cosi2)
			snr_times_D = DH_times_8 * numpy.dot(fpfc2, fpfc_factors)**0.5

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
			try:
				start_index = snr_sequence[max_snr_times_D / (snr_times_D.min() / snr_min)]
			except ZeroDivisionError:
				# one of the instruments that must be able
				# to see the event is blind to it
				continue

			# min_D_other is minimum distance at which source
			# becomes visible in an instrument that isn't
			# involved.  max_snr_times_D / min_D_other gives
			# the SNR in the most sensitive instrument at which
			# the source becomes visible to one of the
			# instruments not allowed to participate
			if len(DH_times_8_other):
				min_D_other = (DH_times_8_other * numpy.dot(fpfc2_other, fpfc_factors)**0.5).min() / cls.snr_min
				try:
					end_index = snr_sequence[max_snr_times_D / min_D_other] + 1
				except ZeroDivisionError:
					# all instruments that must not see
					# it are blind to it
					end_index = None
			else:
				# there are no other instruments
				end_index = None

			# if start_index >= end_index then in order for the
			# source to be close enough to be visible in all
			# the instruments that must see it it is already
			# visible to one or more instruments that must not.
			# don't need to check for this, the for loop that
			# comes next will simply not have any iterations.

			# iterate over the nominal SNRs (= noise-free SNR
			# in the most sensitive instrument) at which we
			# will add weight to the PDF.  from the SNR in
			# most sensitive instrument, the distance to the
			# source is:
			#
			#	D = max_snr_times_D / snr
			#
			# and the (noise-free) SNRs in all instruments are:
			#
			#	snr_times_D / D
			#
			# scipy's Rice-distributed RV code is used to
			# add the effect of background noise, converting
			# the noise-free SNRs into simulated observed SNRs
			#
			# number of sources b/w Dlo and Dhi:
			#
			#	d count \propto D^2 |dD|
			#	  count \propto Dhi^3 - Dlo**3
			D_Dhi_Dlo_sequence = max_snr_times_D / snr_snrlo_snrhi_sequence[start_index:end_index]
			for snr, weight in zip(rice_rvs(snr_times_D / numpy.reshape(D_Dhi_Dlo_sequence[:,0], (len(D_Dhi_Dlo_sequence), 1))), D_Dhi_Dlo_sequence[:,1]**3. - D_Dhi_Dlo_sequence[:,2]**3.):
				pdf[tuple(snr)] += weight

			if progressbar is not None:
				progressbar.increment()
		# check for divide-by-zeros that weren't caught.  also
		# finds NaNs if they're there
		assert numpy.isfinite(pdf.array).all()

		# convolve the samples with a Gaussian density estimation
		# kernel
		rate.filter_array(pdf.array, rate.gaussian_window(*(1.875,) * len(pdf.array.shape)))
		# protect against round-off in FFT convolution leading to
		# negative values in the PDF
		numpy.clip(pdf.array, 0., PosInf, pdf.array)
		# zero counts in bins that are below the trigger threshold.
		# have to convert SNRs to indexes ourselves and adjust so
		# that we don't zero the bin in which the SNR threshold
		# falls
		range_all = slice(None, None)
		range_low = slice(None, pdf.bins[0][cls.snr_min])
		for i in xrange(len(instruments)):
			slices = [range_all] * len(instruments)
			slices[i] = range_low
			pdf.array[slices] = 0.
		# convert bin counts to normalized PDF
		pdf.to_pdf()
		# one last sanity check
		assert numpy.isfinite(pdf.array).all()
		# done
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

	if n_samples < 1:
		raise ValueError("n_samples=%d must be >= 1" % n_samples)
	if min_distance < 0.:
		raise ValueError("min_distance=%g must be >= 0" % min_distance)

	# get instrument names
	names = tuple(horizon_history)
	if not names:
		raise ValueError("horizon_history is empty")
	# get responses in that same order
	resps = [lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in names]

	# initialize output.  dictionary mapping instrument combination to
	# probability (initially all 0).
	result = dict.fromkeys((frozenset(instruments) for n in xrange(2, len(names) + 1) for instruments in iterutils.choices(names, n)), 0.0)

	# we select random uniformly-distributed right ascensions so
	# there's no point in also choosing random GMSTs and any value is
	# as good as any other
	gmst = 0.0

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
	xlal_am_resp = lal.ComputeDetAMResponse

	# loop as many times as requested
	for i in xrange(n_samples):
		# retrieve random horizon distances in the same order as
		# the instruments.  note:  rand_horizon_distances() is only
		# evaluated once in this expression.  that's important
		DH = numpy_array(map(rand_horizon_distances().__getitem__, names))

		# select random sky location and source orbital plane
		# inclination and choice of polarization
		theta = acos(random_uniform(-1., 1.))
		phi = random_uniform(0., twopi)
		psi = random_uniform(0., twopi)
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
		#
		# NOTE:  noise-induced SNR fluctuations have the effect of
		# allowing sources slightly farther away than would
		# nominally allow them to be detectable to be seen above
		# the detection threshold with some non-zero probability,
		# and sources close enough to be detectable to be masked by
		# noise and missed with some non-zero probability.
		# accounting for this effect correctly shows it to provide
		# an additional multiplicative factor to the volume that
		# depends only on the SNR threshold.  therefore, like all
		# the other factors common to all instruments, it too can
		# be ignored.
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
	return signal_rates, noise_rates


def binned_log_likelihood_ratio_rates_from_samples_wrapper(queue, *args, **kwargs):
	try:
		queue.put(binned_log_likelihood_ratio_rates_from_samples(*args, **kwargs))
	except:
		queue.put(None)
		raise


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
		"ln_likelihood_ratio": rate.NDBins((rate.ATanBins(0., 110., 3000),))
	}

	filters = {
		"ln_likelihood_ratio": rate.gaussian_window(8.)
	}

	#
	# Threshold at which FAP & FAR normalization will occur
	#

	ln_likelihood_ratio_threshold = NegInf


	def __init__(self, coinc_params_distributions, instruments, process_id = None, nsamples = 1000000, verbose = False):
		self.background_likelihood_rates = {}
		self.background_likelihood_pdfs = {}
		self.signal_likelihood_rates = {}
		self.signal_likelihood_pdfs = {}
		self.zero_lag_likelihood_rates = {}
		self.zero_lag_likelihood_pdfs = {}
		self.process_id = process_id

		#
		# initialize binnings
		#

		instruments = tuple(instruments)
		for key in [frozenset(ifos) for n in range(2, len(instruments) + 1) for ifos in iterutils.choices(instruments, n)]:
			self.background_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])
			self.signal_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])
			self.zero_lag_likelihood_rates[key] = rate.BinnedArray(self.binnings["ln_likelihood_ratio"])

		#
		# bailout out used by .from_xml() class method to get an
		# uninitialized instance
		#

		if coinc_params_distributions is None:
			return

		#
		# run importance-weighted random sampling to populate
		# binnings.  one thread per instrument combination
		#

		threads = []
		for key in self.background_likelihood_rates:
			if verbose:
				print >>sys.stderr, "computing ranking statistic PDFs for %s" % ", ".join(sorted(key))
			q = multiprocessing.queues.SimpleQueue()
			p = multiprocessing.Process(target = lambda: binned_log_likelihood_ratio_rates_from_samples_wrapper(
				q,
				self.signal_likelihood_rates[key],
				self.background_likelihood_rates[key],
				snglcoinc.LnLikelihoodRatio(coinc_params_distributions).samples(coinc_params_distributions.random_params(key)),
				nsamples = nsamples
			))
			p.start()
			threads.append((p, q, key))
		while threads:
			p, q, key = threads.pop(0)
			self.signal_likelihood_rates[key], self.background_likelihood_rates[key] = q.get()
			p.join()
			if p.exitcode:
				raise Exception("sampling thread failed")
		if verbose:
			print >>sys.stderr, "done computing ranking statistic PDFs"

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
			self.zero_lag_likelihood_rates[frozenset(lsctables.instrument_set_from_ifos(instruments))][ln_likelihood_ratio,] += 1.

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
		self = cls(None, (), process_id = ligolw_param.get_pyvalue(xml, u"process_id"))

		# pull out the likelihood count and PDF arrays
		def reconstruct(xml, prefix, target_dict):
			for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and ("_%s" % prefix) in elem.Name]:
				ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.Name.split("_")[0]))
				target_dict[ifo_set] = rate.BinnedArray.from_xml(ba_elem, ba_elem.Name.split(":")[0])
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

		# pull out both the bg counts and the pdfs, we need 'em both
		bgcounts_ba = ranking_stats.background_likelihood_rates[None]
		bgpdf_ba = ranking_stats.background_likelihood_pdfs[None]
		# we also need the zero lag counts to build the extinction model
		zlagcounts_ba = ranking_stats.zero_lag_likelihood_rates[None]

		# safety checks
		assert not numpy.isnan(bgcounts_ba.array).any(), "log likelihood ratio rates contains NaNs"
		assert not (bgcounts_ba.array < 0.0).any(), "log likelihood ratio rate contains negative values"
		assert not numpy.isnan(bgpdf_ba.array).any(), "log likelihood ratio pdf contains NaNs"
		assert not (bgpdf_ba.array < 0.0).any(), "log likelihood ratio pdf contains negative values"

		# grab bins that are not infinite in size
		finite_bins = numpy.isfinite(bgcounts_ba.bins.volumes())
		ranks = bgcounts_ba.bins.upper()[0].compress(finite_bins)
		drank = bgcounts_ba.bins.volumes().compress(finite_bins)

		# whittle down the arrays of counts and pdfs
		bgcounts_ba_array = bgcounts_ba.array.compress(finite_bins)
		bgpdf_ba_array = bgpdf_ba.array.compress(finite_bins)
		zlagcounts_ba_array = zlagcounts_ba.array.compress(finite_bins)

		# get the extincted background PDF
		self.zero_lag_total_count = zlagcounts_ba_array.sum()
		extinct_bf_pdf = self.extinct(bgcounts_ba_array, bgpdf_ba_array, zlagcounts_ba_array, ranks)

		# cumulative distribution function and its complement.
		# it's numerically better to recompute the ccdf by
		# reversing the array of weights than trying to subtract
		# the cdf from 1.
		weights = extinct_bf_pdf * drank
		cdf = weights.cumsum()
		cdf /= cdf[-1]
		ccdf = weights[::-1].cumsum()[::-1]
		ccdf /= ccdf[0]

		# cdf boundary condition:  cdf = 1/e at the ranking
		# statistic threshold so that self.far_from_rank(threshold)
		# * livetime = observed count of events above threshold.
		# FIXME this doesn't actually work.
		# FIXME not doing it doesn't actually work.
		# ccdf *= 1. - 1. / math.e
		# cdf *= 1. - 1. / math.e
		# cdf += 1. / math.e

		# last checks that the CDF and CCDF are OK
		assert not numpy.isnan(cdf).any(), "log likelihood ratio CDF contains NaNs"
		assert not numpy.isnan(ccdf).any(), "log likelihood ratio CCDF contains NaNs"
		assert ((0. <= cdf) & (cdf <= 1.)).all(), "log likelihood ratio CDF failed to be normalized"
		assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "log likelihood ratio CCDF failed to be normalized"
		assert (abs(1. - (cdf[:-1] + ccdf[1:])) < 1e-12).all(), "log likelihood ratio CDF + CCDF != 1 (max error = %g)" % abs(1. - (cdf[:-1] + ccdf[1:])).max()

		# build interpolators
		self.cdf_interpolator = interpolate.interp1d(ranks, cdf)
		self.ccdf_interpolator = interpolate.interp1d(ranks, ccdf)

		# record min and max ranks so we know which end of the ccdf
		# to use when we're out of bounds
		self.minrank = min(ranks)
		self.maxrank = max(ranks)

	def extinct(self, bgcounts_ba_array, bgpdf_ba_array, zlagcounts_ba_array, ranks):
		# Generate arrays of complementary cumulative counts
		# for background events (monte carlo, pre clustering)
		# and zero lag events (observed, post clustering)
		zero_lag_compcumcount = zlagcounts_ba_array[::-1].cumsum()[::-1]
		bg_compcumcount = bgcounts_ba_array[::-1].cumsum()[::-1]

		# Fit for the number of preclustered, independent coincs by
		# only considering the observed counts safely in the bulk of
		# the distribution.  Only do the fit above 10 counts and below
		# 10000, unless that can't be met and trigger a warning
		fit_min_rank = 1.
		fit_min_counts = min(10., self.zero_lag_total_count / 10. + 1)
		fit_max_counts = min(10000., self.zero_lag_total_count / 10. + 2) # the +2 gaurantees that fit_max_counts > fit_min_counts
		rank_range = numpy.logical_and(ranks > fit_min_rank, numpy.logical_and(zero_lag_compcumcount < fit_max_counts, zero_lag_compcumcount > fit_min_counts))
		if fit_max_counts < 10000.:
			warnings.warn("There are less than 100000 coincidences, extinction effects on background may not be accurately calculated, which will decrease the accuracy of the combined instruments background estimation.")
		if zero_lag_compcumcount.compress(rank_range).size < 1:
			raise ValueError("not enough zero lag data to fit background")

		# Use curve fit to find the predicted total preclustering
		# count. First we need an interpolator of the counts
		obs_counts = interpolate.interp1d(ranks, bg_compcumcount)
		bg_pdf_interp = interpolate.interp1d(ranks, bgpdf_ba_array)

		def extincted_counts(x, N_ratio):
			out = max(zero_lag_compcumcount) * (1. - numpy.exp(-obs_counts(x) * N_ratio))
			out[~numpy.isfinite(out)] = 0.
			return out

		def extincted_pdf(x, N_ratio):
			out = numpy.exp(numpy.log(N_ratio) - obs_counts(x) * N_ratio + numpy.log(bg_pdf_interp(x)))
			out[~numpy.isfinite(out)] = 0.
			return out

		# Fit for the ratio of unclustered to clustered triggers.
		# Only fit N_ratio over the range of ranks decided above
		precluster_normalization, precluster_covariance_matrix = curve_fit(
			extincted_counts,
			ranks[rank_range],
			zero_lag_compcumcount.compress(rank_range),
			sigma = zero_lag_compcumcount.compress(rank_range)**.5,
			p0 = 1e-4
		)

		N_ratio = precluster_normalization[0]

		return extincted_pdf(ranks, N_ratio)

	def fap_from_rank(self, rank):
		# implements equation (8) from Phys. Rev. D 88, 024025.
		# arXiv:1209.0718
		rank = max(self.minrank, min(self.maxrank, rank))
		fap = float(self.ccdf_interpolator(rank))
		return fap_after_trials(fap, self.zero_lag_total_count)

	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.  the return value is divided by T to
		# convert events/experiment to events/second.
		assert self.livetime is not None, "cannot compute FAR without livetime"
		rank = max(self.minrank, min(self.maxrank, rank))
		# true-dismissal probability = 1 - single-event false-alarm
		# probability, the integral in equation (B4)
		tdp = float(self.cdf_interpolator(rank))
		try:
			log_tdp = math.log(tdp)
		except ValueError:
			# TDP = 0 --> FAR = +inf
			return PosInf
		if log_tdp >= -1e-9:
			# rare event:  avoid underflow by using log1p(-FAP)
			log_tdp = math.log1p(-float(self.ccdf_interpolator(rank)))
		return self.zero_lag_total_count * -log_tdp / self.livetime

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


def get_live_time(seglistdict, verbose = False):
	livetime = float(abs(vote((segs for instrument, segs in seglistdict.items() if instrument != "H2"), 2)))
	if verbose:
		print >> sys.stderr, "Livetime: %.3g s" % livetime
	return livetime


def get_live_time_segs_from_search_summary_table(connection, program_name = "gstlal_inspiral"):
	xmldoc = dbtables.get_xml(connection)
	farsegs = ligolw_search_summary.segmentlistdict_fromsearchsummary(xmldoc, program_name).coalesce()
	xmldoc.unlink()
	return farsegs
