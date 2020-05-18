# Copyright (C) 2020 Daichi Tsuna 
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
import scipy
from scipy import stats

from ligo.lw import array as ligolw_array
from gstlal import stats as gstlalstats
import lal
from lal import rate
import lalsimulation


__doc__ = """

The goal of this module is to implement the probability of getting a given set
of extrinsic parameters for a set of detectors parameterized by n-tuples of
trigger parameters:  assuming that the event is a gravitational wave
signal, *s*, coming from an isotropic distribution in location, orientation and
the volume of space.  The implementation of this in the calling code can be 
found in :py:mod:`string_lr_far`.
"""


def SNRPDF(instruments, horizon_distances, snr_cutoff, n_samples = 200000, bins = rate.ATanLogarithmicBins(3.6, 1e3, 150)):
	"""
	Precomputed SNR PDF for each detector
	Returns a BinnedArray containing
	P(snr_{inst1}, snr_{inst2}, ... | signal seen in exactly
		{inst1, inst2, ...} in a network of instruments
		with a given set of horizon distances)
	
	i.e., the joint probability density of observing a set of
	SNRs conditional on them being the result of signal that
	has been recovered in a given subset of the instruments in
	a network of instruments with a given set of horizon
	distances.

	The axes of the PDF correspond to the instruments in
	alphabetical order.  The binning used for all axes is set
	with the bins parameter.

	The n_samples parameter sets the number of iterations for
	the internal Monte Carlo sampling loop.
	"""
	if n_samples < 1:
		raise ValueError("n_samples=%d must be >= 1" % n_samples)

	# get instrument names
	instruments = sorted(instruments)
	if len(instruments) < 1:
		raise ValueError(instruments)
	# get the horizon distances in the same order
	DH_times_8 = 8. * numpy.array([horizon_distances[inst] for inst in instruments])
	# get detector responses in the same order
	resps = [lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in instruments]

	# get horizon distances and responses of remaining
	# instruments (order doesn't matter as long as they're in
	# the same order)
	DH_times_8_other = 8. * numpy.array([dist for inst, dist in horizon_distances.items() if inst not in instruments])
	resps_other = tuple(lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in horizon_distances if inst not in instruments)

	# initialize the PDF array, and pre-construct the sequence of
	# snr, d(snr) tuples. since the last SNR bin probably has
	# infinite size, we remove it from the sequence
	# (meaning the PDF will be left 0 in that bin)
	pdf = rate.BinnedArray(rate.NDBins([bins] * len(instruments)))
	snr_sequence = rate.ATanLogarithmicBins(3.6, 1e3, 500)
	snr_snrlo_snrhi_sequence = numpy.array(zip(snr_sequence.centres(), snr_sequence.lower(), snr_sequence.upper())[:-1])

	# compute the SNR at which to begin iterations over bins
	assert type(snr_cutoff) is float
	snr_min = snr_cutoff - 3.0
	assert snr_min > 0.0

	# we select random uniformly-distributed right assensions
	# so there's no point in also choosing random GMSTs and any
	# vlaue is as good as any other
	gmst = 0.0

	# run the sample the requested # of iterations. save some
	# symbols to avoid doing module attribute look-ups in the
	# loop
	acos = math.acos
	random_uniform = random.uniform
	twopi = 2. * math.pi
	pi_2 = math.pi / 2.
	xlal_am_resp = lal.ComputeDetAMResponse
	# FIXME:  scipy.stats.rice.rvs broken on reference OS.
	# switch to it when we can rely on a new-enough scipy
	#rice_rvs = stats.rice.rvs	# broken on reference OS
	# the .reshape is needed in the event that x is a 1x1
	# array:  numpy returns a scalar from sqrt(), but we must
	# have something that we can iterate over
	rice_rvs = lambda x: numpy.sqrt(stats.ncx2.rvs(2., x**2.)).reshape(x.shape)

	for i in xrange(n_samples):
		# select random sky location and source orbital
		# plane inclination
		# the signal is linearly polaraized, and h_cross = 0
		# is assumed, so we need only F+ (its absolute value).
		ra = random_uniform(0., twopi)
		dec = pi_2 - acos(random_uniform(-1., 1.))
		psi = random_uniform(0., twopi)
		fplus = tuple(abs(xlal_am_resp(resp, ra, dec, psi, gmst)[0]) for resp in resps)

		# 1/8 ratio of inverse SNR to distance for each instrument
		# (1/8 because horizon distance is defined for an SNR of 8,
		# and we've omitted that factor for performance)
		snr_times_D = DH_times_8 * fplus 

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
			min_D_other = (DH_times_8_other * fplus).min() / snr_cutoff
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

	# check for divide-by-zeros that weren't caught.  also
	# finds nans if they are there
	assert numpy.isfinite(pdf.array).all()

	# convolve samples with gaussian kernel
	rate.filter_array(pdf.array, rate.gaussian_window(*(1.875,) * len(pdf.array.shape)))
	# protect against round-off in FFT convolution leading to
	# negative valuesin the PDF
	numpy.clip(pdf.array, 0., PosInf, pdf.array)
	# zero counts in bins that are below the trigger threshold.
	# have to convert SNRs to indexes ourselves and adjust so
	# that we don't zero the bin in which the SNR threshold
	# falls
	range_all = slice(None, None)
	range_low = slice(None, pdf.bins[0][snr_cutoff])
	for i in xrange(len(instruments)):
		slices = [range_all] * len(instruments)
		slices[i] = range_low
	# convert bin counts to normalized PDF
	pdf.to_pdf()
	# one last sanity check
	assert numpy.isfinite(pdf.array).all()
	# done
	return pdf


def P_instruments_given_signal(horizon_distances, n_samples = 500000, min_instruments = 2):
	"""
	Precomputed P(ifos | {horizon distances}, signal).
	Returns a dictionary containing probabilities of each instrument producing a trigger
	given the instruments' horizon distances.
	"""
	if n_samples < 1:
		raise ValueError("n_samples=%d must be >= 1" % n_samples)
	if min_instruments < 1:
		raise ValueError("min_instruments=%d must be >= 1" % min_instruments)
	# get instrument names
	names = tuple(horizon_distances.keys())
	# get the horizon distances in the same order
	DH = numpy.array(tuple(horizon_distances.values()))
	# get detector responses in the same order
	resps = [lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in names]

	# initialize output.
	result = dict.fromkeys((frozenset(instruments) for n in range(min_instruments, len(names) + 1) for instruments in itertools.combinations(names, n)), 0.0)
	if not result:
		raise ValueError("not enough instruments in horizon_distances to satisfy min_instruments")

	# check for no-op
	if (DH != 0.).sum() < min_instruments:
		# not enough instruments are on to form a coinc with
		# the minimum required instruments. this is not
		# considered an error condition, returns p=0 for all
		# probabilities. NOTE result is not normalizable
		return result

	# we select random uniformly-distributed right assensions
	# so there's no point in also choosing random GMSTs and any
	# vlaue is as good as any other
	gmst = 0.0

	# in the loop, we'll need a sequence of integers to enumerate
	# instruments. construct it here to avoid doing it repeatedly in
	# the loop
	indices = tuple(range(len(names)))

	# run the sample the requested # of iterations. save some
	# symbols to avoid doing module attribute look-ups in the
	# loop
	acos = math.acos
	random_uniform = random.uniform
	twopi = 2. * math.pi
	pi_2 = math.pi / 2.
	xlal_am_resp = lal.ComputeDetAMResponse

	for i in xrange(n_samples):
		# select random sky location and source orbital
		# plane inclination
		# the signal is linearly polaraized, and h_cross = 0
		# is assumed, so we need only F+ (its absolute value)
		ra = random_uniform(0., twopi)
		dec = pi_2 - acos(random_uniform(-1., 1.))
		psi = random_uniform(0., twopi)
		fplus = tuple(abs(xlal_am_resp(resp, ra, dec, psi, gmst)[0]) for resp in resps)

		# 1/8 ratio of inverse SNR to distance for each instrument
		# (1/8 because horizon distance is defined for an SNR of 8,
		# and we've omitted that factor for performance)
		snr_times_D_over_8 = DH * fplus 

		# the volume visible to each instrument given the
		# requirement that a source be above the SNR threshold is
		#
		# V = [constant] *(8 * snr_times_D_over_8 / snr_thresh)**3
		#
		# but in the end we'll only need ratio of these volumes, so
		# we can omit the proportionality constant anad we can also
		# omit the factor of (8/snr_thresh)**3.
		# NOTE this assumes all the detectors have same SNR threshold
		V_at_snr_threshold = snr_times_D_over_8**3.

		# order[0] is the index of instrument that can see sources the
		# farthest, order[1] is index of instrument that can see
		# sources the next farthest, ...
		order = sorted(indices, key = V_at_snr_threshold.__getitem__, reverse = True)
		ordered_names = tuple(names[i] for i in order)

		# instrument combination and volume of space (up to
		# irrelevant proportionality constant) visible to that
		# combination given the requirement that a source be above
		# the SNR threshold in that combination. sequence of
		# instrument combinations is left as a generator expression
		# for lazy evaluation
		instruments = (frozenset(ordered_names[:n]) for n in xrange(min_instruments, len(order) + 1))
		V = tuple(V_at_snr_threshold[i] for i in order[min_instruments-1:])
		
		# for each instrument combination, probability that a
		# source visible to at least the minimum required number of
		# instruments is visible to that combination (here is where
		# the proportionality constant and factor (8/snr_threshold)**3
		# drop out of the calculation
		P = tuple(x / V[0] for x in V)

		# accumulate result. p - pnext is the probability that a
		# source (that is visible to at least the minimum required
		# number of instruments) is visible to that combination of
		# instruments and not any other combination of instruments
		for key, p, pnext in zip(instruments, P, P[1:] + (0.,)):
			result[key] += p - pnext

	# normalize
	for key in result:
		result[key] /= n_samples

	#
	# make sure it's normalized. allow an all-0 result in the event
	# that too few instruments are available to ever form coincs.
	# 

	total = sum(sorted(result.values()))
	assert abs(total - 1.) < 1e-13 or total == 0., "result insufficiently well normalized: %s, sum = %g" % (result, total)
	if total != 0:
		for key in result:
			result[key] /= total

	#
	# done
	#

	return result
