# Copyright (C) 2016  Kipp Cannon
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


import math
import numpy
import random
import sys


from glue import iterutils
import lal
import lalsimulation
from pylal import rate
from pylal import snglcoinc


__all__ = [
	"instruments_rate_given_noise",
	"P_instruments_given_signal",
	"SNRPDF"
]


#
# =============================================================================
#
#                              Instrument Combos
#
# =============================================================================
#


def instruments_rate_given_noise(singles_counts, zero_lag_coinc_counts, segs, delta_t, min_instruments = 2, verbose = False):
	"""
	Models the expected number of background coincidence events.  Most
	of the work is performed by pylal.snglcoinc.CoincSynthesizer.  The
	output of that code is corrected using the assumption that the
	events whose total counts are provided as input to this function
	are occuring in distinct "bins" (= templates) and that they can
	only form a coincidence if they are in the same bin.  The number of
	distinct bins is solved for using the actual observed zero-lag
	coincidence counts.

	Note the odd cart-before-the-horse logic:  we need to know the
	number of observed coincidence events in order to predict the
	number of observed coincidence events.  However, the observed event
	counts are used *only* to solve for the number of distinct bins.

	Example:

	>>> from glue.segments import *
	>>> singles_counts = {"H1": 33, "L1": 35, "V1": 55}
	>>> zero_lag_coinc_counts = {frozenset(['V1', 'H1']): 1, frozenset(['V1', 'H1', 'L1']): 0, frozenset(['H1', 'L1']): 1, frozenset(['V1', 'L1']): 2}
	>>> seglists = segmentlistdict({"H1": segmentlist([segment(0, 30)]), "L1": segmentlist([segment(10, 50)]), "V1": segmentlist([segment(20, 70)])})
	>>> instruments_rate_given_noise(singles_counts, zero_lag_coinc_counts, seglists, 0.005)
	{frozenset(['V1', 'H1']): 0.7635015339606976, frozenset(['V1', 'H1', 'L1']): 0.017867580438332864, frozenset(['H1', 'L1']): 0.5601269964222882, frozenset(['V1', 'L1']): 1.7982741132776223}
	"""
	if set(singles_counts) != set(segs):
		raise ValueError("singles_counts (%s) and segs (%s) must be for the same instruments" % (", ".join(sorted(singles_counts)), ", ".join(sorted(segs))))

	#
	# initialize the CoincSynthesizer object
	#

	if verbose:
		print >>sys.stderr, "synthesizing background-like instrument combination probabilities ..."
	coincsynth = snglcoinc.CoincSynthesizer(
		eventlists = singles_counts,
		segmentlists = segs,
		delta_t = delta_t,
		min_instruments = min_instruments
	)

	if set(zero_lag_coinc_counts) < set(coincsynth.all_instrument_combos):
		raise ValueError("zero_lag_coinc_counts must provide a count for each possible instrument combo:  missing %s" % ", ".join(sorted(set(coincsynth.all_instrument_combos) - set(zero_lag_coinc_counts))))

	# assume the single-instrument events are being collected in
	# several disjoint bins so that events from different instruments
	# that occur at the same time but in different bins are not
	# coincident.  if there are M bins for each instrument, the
	# probability that N events all occur in the same bin is
	# (1/M)^(N-1).  the number of bins, M, is therefore given by the
	# (N-1)th root of the ratio of the predicted number of N-instrument
	# coincs to the observed number of N-instrument coincs.  use the
	# average of M measured from all instrument combinations.
	#
	# finding M by comparing predicted to observed zero-lag counts
	# assumes we are in a noise-dominated regime, i.e., that the
	# observed relative abundances of coincs are not significantly
	# contaminated by signals.  if signals are present in large
	# numbers, and occur in different abundances than the noise events,
	# averaging the apparent M over different instrument combinations
	# helps to suppress the contamination.  NOTE:  the number of
	# coincidence bins, M, should be very close to the number of
	# templates
	N = 0
	coincidence_bins = 0.
	for instruments in coincsynth.all_instrument_combos:
		if len(instruments) < 2:
			# this calculation only depends on the counts of
			# genuine coincs, not single-instrument "coincs" if
			# they are allowed.
			continue
		predicted_count = coincsynth.mean_coinc_count[instruments]
		observed_count = zero_lag_coinc_counts[instruments]
		if predicted_count > 0 and observed_count > 0:
			coincidence_bins += (predicted_count / observed_count)**(1. / (len(instruments) - 1))
			N += 1
	assert N > 0
	assert coincidence_bins > 0.
	coincidence_bins /= N
	if verbose:
		print >>sys.stderr, "\tthere seems to be %g effective disjoint coincidence bin(s).  using %g" % (coincidence_bins, round(coincidence_bins))
	coincidence_bins = round(coincidence_bins)

	# assume the rate is uniform across bins to convert observed
	# single-instrument event rates to rates/bin by dividing by the
	# number of bins
	coincsynth.mu = dict((instrument, rate / coincidence_bins) for instrument, rate in coincsynth.mu.items())

	# now compute the expected coincidence rates/bin/instrument combo,
	# then multiply by the number of bins to get the expected
	# coincidence rate/instrument combination.
	return dict((instruments, count * coincidence_bins) for instruments, count in coincsynth.mean_coinc_count.items())


def P_instruments_given_signal(horizon_history, n_samples = 500000, min_instruments = 2, min_distance = 0.):
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
	if min_instruments < 1:
		raise ValueError("min_instruments=%d must be >= 1" % min_instruments)

	# get instrument names
	names = tuple(horizon_history)
	if not names:
		raise ValueError("horizon_history is empty")
	# get responses in that same order
	resps = [lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in names]

	# initialize output.  dictionary mapping instrument combination to
	# probability (initially all 0).
	result = dict.fromkeys((frozenset(instruments) for n in xrange(min_instruments, len(names) + 1) for instruments in iterutils.choices(names, n)), 0.0)
	if not result:
		raise ValueError("not enough instruments in horizon_history to satisfy min_instruments")

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
	successes = fails = 0
	while successes < n_samples:
		# retrieve random horizon distances in the same order as
		# the instruments.  note:  rand_horizon_distances() is only
		# evaluated once in this expression.  that's important
		DH = numpy_array(map(rand_horizon_distances().__getitem__, names))

		# select random sky location and source orbital plane
		# inclination and choice of polarization
		ra = random_uniform(0., twopi)
		dec = pi_2 - acos(random_uniform(-1., 1.))
		psi = random_uniform(0., twopi)
		cosi2 = random_uniform(-1., 1.)**2.

		# compute F+^2 and Fx^2 for each antenna from the sky
		# location and antenna responses
		fpfc2 = numpy_array(tuple(xlal_am_resp(resp, ra, dec, psi, gmst) for resp in resps))**2.

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
		instruments = (frozenset(ordered_names[:n]) for n in xrange(min_instruments, len(order) + 1))
		V = tuple(V_at_snr_threshold[i] for i in order[min_instruments - 1:])
		if V[0] <= min_distance:
			# fewer than the required minimum number of
			# instruments are on, so no combination can see
			# anything
			fails += 1
			if successes + fails >= n_samples and fails / float(successes + fails) > .90:
				raise ValueError("convergence too slow:  success/fail ratio = %d/%d" % (successes, fails))
			continue

		# for each instrument combination, probability that a
		# source visible to at least the minimum required number of
		# instruments is visible to that combination (here is where
		# the proportionality constant and factor of
		# (8/snr_threshold)**3 drop out of the calculation)
		P = tuple(x / V[0] for x in V)

		# accumulate result.  p - pnext is the probability that a
		# source (that is visible to at least the minimum required
		# number of instruments) is visible to that combination of
		# instruments and not any other combination of instruments.
		for key, p, pnext in zip(instruments, P, P[1:] + (0.,)):
			result[key] += p - pnext
		successes += 1
	for key in result:
		result[key] /= n_samples

	#
	# make sure it's normalized
	#

	total = sum(sorted(result.values()))
	assert abs(total - 1.) < 1e-13, "result insufficiently well normalized: %s, sum = %g" % (result, total)
	for key in result:
		result[key] /= total

	#
	# done
	#

	return result


#
# =============================================================================
#
#                                   SNR PDF
#
# =============================================================================
#


class SNRPDF(object):
	@staticmethod
	def joint_pdf_of_snrs(instruments, inst_horiz_mapping, snr_cutoff, n_samples = 160000, bins = rate.ATanLogarithmicBins(3.6, 1200., 170), progressbar = None):
		"""
		Return a BinnedArray containing

		P(snr_{inst1}, snr_{inst2}, ... | signal seen in exactly
			{inst1, inst2, ...} in a network of instruments
			with a given set of horizon distances)

		i.e., the joint probability density of observing a set of
		SNRs conditional on them being the result of signal that
		has been recovered in a given subset of the instruments in
		a network of instruments with a given set of horizon
		distances.

		The snr_cutoff parameter sets the minimum SNR required for
		a trigger (it is assumed SNRs below this value are
		impossible to observe).

		The instruments parameter specifies the instruments upon
		whose triggers the SNR distribution is conditional --- the
		SNR distribution is conditional on the signal being
		recovered by exactly these instruments and no others.

		The inst_horiz_mapping parameter is a dictionary mapping
		instrument name (e.g., "H1") to horizon distance (arbitrary
		units).  For consistency, all instruments in the network
		must be listed in inst_horiz_mapping regardless of which
		instruments are operating and regardless of which
		instruments the probability density is conditional on the
		signal being recovered in; instruments that are included in
		the network but not operating at the time should be listed
		with a horizon distance of 0.

		The axes of the PDF correspond to the instruments in
		alphabetical order.  The binning used for all axes is set
		with the bins parameter.

		The n_samples parameter sets the number of iterations for
		the internal Monte Carlo sampling loop, and progressbar can
		be a glue.text_progress_bar.ProgressBar instance for
		verbosity.
		"""
		# get instrument names in alphabetical order
		instruments = sorted(instruments)
		if len(instruments) < 1:
			raise ValueError(instruments)
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
		snr_sequence = rate.ATanLogarithmicBins(3.6, 1200., 500)
		snr_snrlo_snrhi_sequence = numpy.array(zip(snr_sequence.centres(), snr_sequence.lower(), snr_sequence.upper())[:-1])

		# compute the SNR at which to begin iterations over bins
		assert type(snr_cutoff) is float
		snr_min = snr_cutoff - 3.0
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
		# the .reshape is needed in the event that x is a 1x1
		# array:  numpy returns a scalar from sqrt(), but we must
		# have something that we can iterate over
		rice_rvs = lambda x: numpy.sqrt(stats.ncx2.rvs(2., x**2.)).reshape(x.shape)
		for i in xrange(n_samples):
			# select random sky location and source orbital
			# plane inclination and choice of polarization
			ra = random_uniform(0., twopi)
			dec = pi_2 - acos(random_uniform(-1., 1.))
			psi = random_uniform(0., twopi)
			cosi2 = random_uniform(-1., 1.)**2.

			# F+^2 and Fx^2 for each instrument
			fpfc2 = numpy.array(tuple(xlal_am_resp(resp, ra, dec, psi, gmst) for resp in resps))**2.
			fpfc2_other = numpy.array(tuple(xlal_am_resp(resp, ra, dec, psi, gmst) for resp in resps_other))**2.

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
				min_D_other = (DH_times_8_other * numpy.dot(fpfc2_other, fpfc_factors)**0.5).min() / snr_cutoff
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
		range_low = slice(None, pdf.bins[0][snr_cutoff])
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
