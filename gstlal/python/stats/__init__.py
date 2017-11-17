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
import numpy
from scipy.special import ive


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
#                                Poisson Stuff
#
# =============================================================================
#


def assert_probability(f):
	def g(*args, **kwargs):
		p = f(*args, **kwargs)
		if isinstance(p, numpy.ndarray):
			assert ((0. <= p) & (p <= 1.)).all()
		else:
			assert 0. <= p <= 1.
		return p
	return g


def assert_ln_probability(f):
	def g(*args, **kwargs):
		p = f(*args, **kwargs)
		if isinstance(p, numpy.ndarray):
			assert (p <= 0.).all()
		else:
			assert p <= 0.
		return p
	return g


@assert_probability
@numpy.vectorize
def poisson_p_not_0(l):
	"""
	Return the probability that a Poisson process with a mean rate of l
	yields a non-zero count.  = 1 - exp(-l).
	"""
	assert l >= 0.

	# need -l everywhere

	l = -l

	#
	# result is closer to 1 than to 0.  use direct evaluation.
	#

	if l < -0.69314718055994529:
		return 1. - math.exp(l)

	#
	# result is closer to 0 than to 1.  use Taylor expansion for exp()
	# with leading term removed to avoid having to subtract from 1 to
	# get answer.
	#
	# 1 - exp x = -(x + x^2/2! + x^3/3! + ...)
	#

	s = [l]
	term = l
	threshold = -1e-20 * l
	for n in itertools.count(2):
		term *= l / n
		s.append(term)
		if abs(term) <= threshold:
			# smallest term was added last, want to compute sum
			# from smallest to largest
			s.reverse()
			s = -sum(s)
			# if the sum was identically 0 then we've ended up
			# with -0.0 which we add a special case for to make
			# positive.
			assert s >= 0.
			return s if s else 0.


@assert_probability
def poisson_p_0(l):
	"""
	Return the probability that a Poisson process with a mean rate of l
	yields a zero count.  = exp(-l), but with a sanity check that l is
	non-negative.
	"""
	assert l >= 0.
	return numpy.exp(-l)


@assert_ln_probability
def poisson_ln_p_0(l):
	"""
	Return the natural logarithm of the probability that a Poisson
	process with a mean rate of l yields a zero count.  = -l, but with
	a sanity check that l is non-negative.
	"""
	assert l >= 0.
	return -l


#
# =============================================================================
#
#                                Binomial Stuff
#
# =============================================================================
#


@assert_probability
@numpy.vectorize
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
	# compute result as 1 - exp(m * log(1 - p)).  use poisson_p_not_0()
	# to evaluate 1 - exp() with an algorithm that avoids loss of
	# precision.
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

	return float(poisson_p_not_0(-x))


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
	assert 0. <= p0 <= 1.	# p0 must be a valid probability
	assert 0. <= p1 <= 1.	# p1 must be a valid probability

	if p0 == 0. or p0 == 1.:
		assert p0 == p1	# require valid relationship
		# but we still can't solve for m
		raise ValueError("m undefined")
	if p1 == 1.:
		return PosInf

	#
	# m log(1 - p0) = log(1 - p1)
	#

	return math.log1p(-p1) / math.log1p(-p0)
