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
import os
import random
from scipy import stats
import sys


from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
from glue.text_progress_bar import ProgressBar
import lal
import lalsimulation
from pylal import rate
from pylal import snglcoinc


# FIXME:  caution, this information might get organized differently later.
# for now we just need to figure out where the gstlal-inspiral directory in
# share/ is.  don't write anything that assumes that this module will
# continue to define any of these symbols
from gstlal import paths as gstlal_config_paths


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


def instruments_rate_given_noise(singles_counts, num_templates, segs, delta_t, min_instruments = 2):
	"""
	Models the expected number of background coincidence events.  Most
	of the work is performed by pylal.snglcoinc.CoincSynthesizer.  The
	output of that code is corrected using the assumption that the
	events whose total counts are provided as input to this function
	are occuring in distinct "bins" (= templates) and that they can
	only form a coincidence if they are in the same bin.

	Example:

	>>> from glue.segments import *
	>>> singles_counts = {"H1": 33000, "L1": 35000, "V1": 55000}
	>>> num_templates = 1000
	>>> seglists = segmentlistdict({"H1": segmentlist([segment(0, 30)]), "L1": segmentlist([segment(10, 50)]), "V1": segmentlist([segment(20, 70)])})
	>>> instruments_rate_given_noise(singles_counts, num_templates, seglists, 0.005, min_instruments = 1)
	{frozenset(['V1']): 52420.355265761136, frozenset(['H1']): 31658.502382616472, frozenset(['V1', 'H1']): 763.5030405229143, frozenset(['L1']): 32623.729803299546, frozenset(['V1', 'L1']): 1798.275619839839, frozenset(['V1', 'H1', 'L1']): 17.866073876116275, frozenset(['H1', 'L1']): 560.128502984505}

	Works for zero counts

	>>> singles_counts = {"H1": 0, "L1": 0, "V1": 0}
	>>> inspiral_extrinsics.instruments_rate_given_noise(singles_counts, num_templates, seglists, 0.005, min_instruments = 1)
	{frozenset(['V1']): 0.0, frozenset(['H1']): 0.0, frozenset(['V1', 'H1']): 0.0, frozenset(['L1']): 0.0, frozenset(['V1', 'L1']): 0.0, frozenset(['V1', 'H1', 'L1']): 0.0, frozenset(['H1', 'L1']): 0.0}
	"""
	if set(singles_counts) != set(segs):
		raise ValueError("singles_counts (%s) and segs (%s) must be for the same instruments" % (", ".join(sorted(singles_counts)), ", ".join(sorted(segs))))

	#
	# convert counts per observation to counts per template per
	# observation
	#

	singles_counts = dict((key, count / float(num_templates)) for key, count in singles_counts.items())

	if max(singles_counts[instrument] / (float(livetime) / delta_t) for instrument, livetime in abs(segs).items()) >= 1:
		raise ValueError("triggers per coincidence window must be << 1")

	#
	# initialize the CoincSynthesizer object
	#

	coincsynth = snglcoinc.CoincSynthesizer(
		eventlists = singles_counts,
		segmentlists = segs,
		delta_t = delta_t,
		min_instruments = min_instruments
	)

	# now compute the expected coincidence rates/template/instrument
	# combo, then multiply by the number of templates to get the
	# expected coincidence rate/instrument combination.

	return dict((instruments, count * num_templates) for instruments, count in coincsynth.mean_coinc_count.items())


def P_instruments_given_signal(horizon_history, n_samples = 500000, min_instruments = 2, min_distance = 0.):
	"""
	Example:

	>>> from gstlal.stats.horizonhistory import *
	>>> hist = HorizonHistories({"H1": NearestLeafTree([(0., 120.)]), "L1": NearestLeafTree([(0., 120.)])})
	>>> P_instruments_given_signal(hist)
	{frozenset(['H1', 'L1']): 1.0}
	>>> P_instruments_given_signal(hist, min_instruments = 1)
	{frozenset(['L1']): 0.25423904879460091, frozenset(['H1', 'L1']): 0.49116512120682387, frozenset(['H1']): 0.25459582999857527}
	"""
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
	result = dict.fromkeys((frozenset(instruments) for n in xrange(min_instruments, len(names) + 1) for instruments in itertools.combinations(names, n)), 0.0)
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
	#
	# cache of pre-computed SNR joint PDFs.  structure is like
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

	DEFAULT_FILENAME = os.path.join(gstlal_config_paths["pkgdatadir"], "inspiral_snr_pdf.xml.gz")
	max_cached_snr_joint_pdfs = float("+inf")
	snr_joint_pdf_cache = {}

	@ligolw_array.use_in
	@ligolw_param.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass


	# FIXME:  is the default choice of distance quantization appropriate?
	def __init__(self, snr_cutoff, log_distance_tolerance = math.log(1.05)):
		"""
		snr_cutoff sets the minimum SNR below which it is
		impossible to obtain a candidate (the trigger SNR
		threshold).
		"""
		self.snr_cutoff = snr_cutoff
		self.log_distance_tolerance = log_distance_tolerance


	def quantize_horizon_distances(self, horizon_distances, min_ratio = 0.1):
		"""
		if two horizon distances, D1 and D2, differ by less than

			| ln(D1 / D2) | <= log_distance_tolerance

		then they are considered to be equal for the purpose of
		recording horizon distance history, generating joint SNR
		PDFs, and so on.  if the smaller of the two is < min_ratio
		* the larger then the smaller is treated as though it were
		0.
		"""
		if any(horizon_distance < 0. for horizon_distance in horizon_distances.values()):
			raise ValueError("%s: negative horizon distance" % repr(horizon_distances))
		horizon_distance_norm = max(horizon_distances.values())
		# check for no-op:  all distances are 0.
		if horizon_distance_norm == 0.:
			return dict.fromkeys(horizon_distances, NegInf)
		# check for no-op:  map all (non-zero) values to 1
		if math.isinf(self.log_distance_tolerance):
			return dict((instrument, 1 if horizon_distance > 0. else NegInf) for instrument, horizon_distance in horizon_distances.items())
		min_distance = min_ratio * horizon_distance_norm
		return dict((instrument, (NegInf if horizon_distance < min_distance else int(round(math.log(horizon_distance / horizon_distance_norm) / self.log_distance_tolerance)))) for instrument, horizon_distance in horizon_distances.items())


	def quantized_horizon_distances(self, quants):
		if math.isinf(self.log_distance_tolerance):
			return dict((instrument, 0. if math.isinf(quant) else 1.) for instrument, quant in quants)
		return dict((instrument, math.exp(quant * self.log_distance_tolerance)) for instrument, quant in quants)


	def snr_joint_pdf_keyfunc(self, instruments, horizon_distances):
		"""
		Internal function defining the key for cache:  two element
		tuple, first element is frozen set of instruments for which
		this is the PDF, second element is frozen set of
		(instrument, horizon distance) pairs for all instruments in
		the network.  horizon distances are normalized to fractions
		of the largest among them and then the fractions aquantized
		to integer powers of a common factor
		"""
		return frozenset(instruments), frozenset(self.quantize_horizon_distances(horizon_distances).items())


	def get_snr_joint_pdf_binnedarray(self, instruments, horizon_distances):
		pdf, binnedarray, age = self.snr_joint_pdf_cache[self.snr_joint_pdf_keyfunc(instruments, horizon_distances)]
		return binnedarray


	def get_snr_joint_pdf(self, instruments, horizon_distances):
		pdf, binnedarray, age = self.snr_joint_pdf_cache[self.snr_joint_pdf_keyfunc(instruments, horizon_distances)]
		return pdf


	def lnP_snrs(self, snrs, horizon_distances):
		# PDF axes are in alphabetical order
		instruments = sorted(snrs)
		lnpdf = self.get_snr_joint_pdf(instruments, horizon_distances)
		return lnpdf(*map(snrs.__getitem__, instruments))


	def add_to_cache(self, instruments, horizon_distances, verbose = False):
		key = self.snr_joint_pdf_keyfunc(instruments, horizon_distances)
		# already in cache?
		if key in self.snr_joint_pdf_cache:
			pdf, binnedarray, age = self.snr_joint_pdf_cache[key]
			return pdf
		# need to build
		if self.snr_joint_pdf_cache:
			age = max(age for ignored, ignored, age in self.snr_joint_pdf_cache.values()) + 1
		else:
			age = 0
		if verbose:
			print >>sys.stderr, "For horizon distances %s" % ", ".join("%s = %.4g Mpc" % item for item in sorted(horizon_distances.items()))
			progressbar = ProgressBar(text = "%s SNR PDF" % ", ".join(sorted(key[0])))
		else:
			progressbar = None
		binnedarray = self.joint_pdf_of_snrs(key[0], self.quantized_horizon_distances(key[1]), self.snr_cutoff, progressbar = progressbar)
		del progressbar
		lnbinnedarray = binnedarray.copy()
		with numpy.errstate(divide = "ignore"):
			lnbinnedarray.array = numpy.log(lnbinnedarray.array)
		pdf = rate.InterpBinnedArray(lnbinnedarray, fill_value = NegInf)
		self.snr_joint_pdf_cache[key] = pdf, binnedarray, age
		# if the cache is full, delete the entry with the smallest
		# age
		while len(self.snr_joint_pdf_cache) > self.max_cached_snr_joint_pdfs:
			del self.snr_joint_pdf_cache[min((age, key) for key, (ignored, ignored, age) in self.snr_joint_pdf_cache.items())[1]]
		if verbose:
			print >>sys.stderr, "%d of %g slots in SNR PDF cache now in use" % (len(self.snr_joint_pdf_cache), self.max_cached_snr_joint_pdfs)
		return pdf


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


	@classmethod
	def from_xml(cls, xml, name = u"generic"):
		name = u"%s:%s" % (name, u"inspiral_snr_pdf")
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		xml, = xml
		self = cls(snr_cutoff = ligolw_param.get_pyvalue(xml, u"snr_cutoff"), log_distance_tolerance = ligolw_param.get_pyvalue(xml, u"log_distance_tolerance"))
		for elem in xml.childNodes:
			if elem.tagName != ligolw.LIGO_LW.tagName:
				continue
			binnedarray = rate.BinnedArray.from_xml(elem, elem.Name.rsplit(u":", 1)[0])

			key = (
				frozenset(lsctables.instrument_set_from_ifos(ligolw_param.get_pyvalue(elem, u"instruments:key"))),
				frozenset((inst.strip(), float(quant) if math.isinf(float(quant)) else int(quant)) for inst, quant in (inst_quant.split(u"=") for inst_quant in ligolw_param.get_pyvalue(elem, u"quantizedhorizons:key").split(u",")))
			)

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


	def to_xml(self, name = u"generic"):
		xml = ligolw.LIGO_LW()
		xml.Name = u"%s:%s" % (name, u"inspiral_snr_pdf")
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"snr_cutoff", self.snr_cutoff))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"log_distance_tolerance", self.log_distance_tolerance))
		for i, (key, (ignored, binnedarray, ignored)) in enumerate(self.snr_joint_pdf_cache.items()):
			elem = xml.appendChild(binnedarray.to_xml(u"%d:pdf" % i))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"instruments:key", lsctables.ifos_from_instrument_set(key[0])))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"quantizedhorizons:key", u",".join(u"%s=%.17g" % inst_quant for inst_quant in sorted(key[1]))))
		return xml


	@classmethod
	def load(cls, fileobj = None, verbose = False):
		if fileobj is None:
			fileobj = open(cls.DEFAULT_FILENAME)
		return cls.from_xml(ligolw_utils.load_fileobj(fileobj, gz = True, contenthandler = cls.LIGOLWContentHandler)[0])

#
# =============================================================================
#
#                                 dt dphi PDF
#
# =============================================================================
#


dt_chebyshev_coeffs_polynomials = []
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([-597.94227757949329, 2132.773853473127, -2944.126306979932, 1945.3033368083029, -603.91576991927593, 70.322754873993347]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([-187.50681052710425, 695.84172327044325, -1021.0688423797938, 744.3266490236075, -272.12853221391498, 35.542404632554508]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([52.128579054466599, -198.32054234352267, 301.34865541080791, -230.8522943433488, 90.780611645135437, -16.310130528927655]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([48.216566165393878, -171.70632176976451, 238.48370471322843, -159.65005032451785, 50.122296925755677, -5.5466740894321367]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([-34.030336093450863, 121.44714070928059, -169.91439486329773, 115.86873916702386, -38.08411813067778, 4.7396784315927816]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([3.4576823675810178, -12.362609260376738, 17.3300203922424, -11.830868787176165, 3.900284020272442, -0.47157315012399248]))
dt_chebyshev_coeffs_polynomials.append(numpy.poly1d([4.0423239651315113, -14.492611904657275, 20.847419746265583, -15.033846689362553, 5.3953159232942216, -0.78132676885883601]))
norm_polynomial = numpy.poly1d([-348550.84040194791, 2288151.9147818103, -6623881.5646601757, 11116243.157047395, -11958335.1384027, 8606013.1361163966, -4193136.6690072878, 1365634.0450674745, -284615.52077054407, 34296.855844416605, -1815.7135263788341])

dt_chebyshev_coeffs = [0]*13


def __dphi_calc_A(combined_snr, delta_t):
        B = -10.840765 * numpy.abs(delta_t) + 1.072866
        M = 46.403738 * numpy.abs(delta_t) - 0.160205
        nu = 0.009032 * numpy.abs(delta_t) + 0.001150
        return (1.0 / (1.0 + math.exp(-B*(combined_snr - M)))**(1.0/nu))


def __dphi_calc_mu(combined_snr, delta_t):
        if delta_t >= 0.0:
                A = 76.234617*delta_t + 0.001639
                B = 0.290863
                C = 4.775688
                return (3.145953 - A* math.exp(-B*(combined_snr-C)))
        elif delta_t < 0.0:
                A = -130.877663*delta_t - 0.002814
                B = 0.31023
                C = 3.184671
                return (3.145953 + A* math.exp(-B*(combined_snr-C)))


def __dphi_calc_kappa(combined_snr, delta_t):
        K = -176.540199*numpy.abs(delta_t) + 7.4387
        B = -7.599585*numpy.abs(delta_t) + 0.215074
        M = -1331.386835*numpy.abs(delta_t) - 35.309173
        nu = 0.012225*numpy.abs(delta_t) + 0.000066
        return (K / (1.0 + math.exp(-B*(combined_snr - M)))**(1.0/nu))


def lnP_dphi_signal(delta_phi, delta_t, combined_snr):
	# Compute von mises parameters
	A_param = __dphi_calc_A(combined_snr, delta_t)
	mu_param = __dphi_calc_mu(combined_snr, delta_t)
	kappa_param = __dphi_calc_kappa(combined_snr, delta_t)
	C_param = 1.0 - A_param

	return math.log(A_param*stats.vonmises.pdf(delta_phi, kappa_param, loc=mu_param) + C_param/(2*math.pi))


def lnP_dt_signal(dt, snr_ratio):
	# FIXME Work out correct model

	# Fits below an snr ratio of 0.33 are not reliable
	if snr_ratio < 0.33:
		snr_ratio = 0.33

	dt_chebyshev_coeffs[0] = dt_chebyshev_coeffs_polynomials[0](snr_ratio)
	dt_chebyshev_coeffs[2] = dt_chebyshev_coeffs_polynomials[1](snr_ratio)
	dt_chebyshev_coeffs[4] = dt_chebyshev_coeffs_polynomials[2](snr_ratio)
	dt_chebyshev_coeffs[6] = dt_chebyshev_coeffs_polynomials[3](snr_ratio)
	dt_chebyshev_coeffs[8] = dt_chebyshev_coeffs_polynomials[4](snr_ratio)
	dt_chebyshev_coeffs[10] = dt_chebyshev_coeffs_polynomials[5](snr_ratio)
	dt_chebyshev_coeffs[12] = dt_chebyshev_coeffs_polynomials[6](snr_ratio)

	return numpy.polynomial.chebyshev.chebval(dt/0.015013, dt_chebyshev_coeffs) - numpy.log(norm_polynomial(snr_ratio))

def lnP_dt_dphi_uniform_H1L1(coincidence_window_extension):
	# FIXME Dont hardcode
	# NOTE This assumes the standard delta t
	return -math.log((snglcoinc.light_travel_time("H1", "L1") + coincidence_window_extension) * (2. * math.pi))


def lnP_dt_dphi_uniform(params, coincidence_window_extension):
	# NOTE Currently hardcoded for H1L1
	# NOTE this is future proofed so that in > 2 ifo scenarios, the
	# appropriate length can be chosen for the uniform dt distribution
	return lnP_dt_dphi_uniform_H1L1(coincidence_window_extension)


def lnP_dt_dphi(params, coincidence_window_extension, model = "noise"):
	# Return P(dt, dphi|{rho_{IFO}}, signal)
	if model == "noise":
		# FIXME Insert actual noise models
		return lnP_dt_dphi_uniform(params["instruments"], coincidence_window_extension)

	elif model == "signal":
		# FIXME Insert actual signal models
		if sorted(params.t_offset.keys()) == ["H1", "L1"]:
			delta_t = float(params.t_offset["H1"] - params.t_offset["L1"])
			delta_phi = (params.coa_phase["H1"] - params.coa_phase["L1"]) % (2*math.pi)
			combined_snr = math.sqrt(params["H1_snr_chi"][0]**2. + params["L1_snr_chi"][0]**2.)
			if params.horizons["H1"] != 0 and params.horizons["L1"] != 0:
				# neither are zero, proceed as normal
				H1_snr_over_horizon = params["H1_snr_chi"][0] / params.horizons["H1"]
				L1_snr_over_horizon = params["L1_snr_chi"][0] / params.horizons["L1"]

			elif params.horizons["H1"] == params.horizons["L1"]:
				# both are zero, treat as equal
				H1_snr_over_horizon = params["H1_snr_chi"][0]
				L1_snr_over_horizon = params["L1_snr_chi"][0]

			else:
				# one of them is zero, treat snr_ratio as 0, which will get changed to 0.33 in lnP_dt_signal
				# FIXME is this the right thing to do?
				return lnP_dt_signal(abs(delta_t), 0.33) + lnP_dphi_signal(delta_phi, delta_t, combined_snr)

			if H1_snr_over_horizon > L1_snr_over_horizon:
				snr_ratio = L1_snr_over_horizon / H1_snr_over_horizon

			else:
				snr_ratio = H1_snr_over_horizon / L1_snr_over_horizon

			return lnP_dt_signal(abs(delta_t), snr_ratio) + lnP_dphi_signal(delta_phi, delta_t, combined_snr)

		else:
			# IFOs != {H1,L1} case, thus just return uniform
			# distribution so that dt/dphi terms dont affect
			# likelihood ratio
			# FIXME Work out general N detector case
			return lnP_dt_dphi_uniform(params["instruments"], coincidence_window_extension)

	else:
		raise ValueError("invalid data model '%s'" % mode)
