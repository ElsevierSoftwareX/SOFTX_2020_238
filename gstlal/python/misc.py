#
# Copyright (C) 2010,2011  Kipp Cannon <kipp.cannon@ligo.org>
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

## @file

## @package misc

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from scipy import optimize
import numpy, scipy
import sys


#
# import all symbols from _misc
#


from _misc import *


#
# =============================================================================
#
#                                    Extras
#
# =============================================================================
#


#
# inverse of cdf_weighted_chisq_P()
#


def cdf_weighted_chisq_Pinv(A, noncent, dof, var, P, lim, accuracy):
	func = lambda x: cdf_weighted_chisq_P(A, noncent, dof, var, x, lim, accuracy) - P
	lo = 0.0
	hi = 1.0
	while func(hi) < 0:
		lo = hi
		hi *= 2
		print(lo, hi, file=sys.stderr)
	return optimize.brentq(func, lo, hi, xtol = accuracy * 4)

#
# Function to compute the threshold at a fixed FAR for weighted \chi^2
#


def max_stat_thresh(coeffs, fap, samp_tol=100.0):
	num = int(samp_tol/ fap)
	out = numpy.zeros(num)
	for c in coeffs: out += c*scipy.randn(num)**2
	out.sort()
	return float(out[-int(samp_tol)])


#
# Function to compute the optimal quadratic statistic coefficients given
# singular values S and a desired signal size amp
#


def ss_coeffs(S, amp=5.5):
	return S**2. / (S**2. + len(S) / amp**2. )
