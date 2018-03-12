# Copyright (C) 2016,2017  Kipp Cannon
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
from scipy import interpolate, fft, ifft
import sys
import h5py

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
from glue.text_progress_bar import ProgressBar
from gstlal import stats as gstlalstats
import lal
from lal import rate
from lal import LIGOTimeGPS
from lalburst import snglcoinc
import lalsimulation


# FIXME:  caution, this information might get organized differently later.
# for now we just need to figure out where the gstlal-inspiral directory in
# share/ is.  don't write anything that assumes that this module will
# continue to define any of these symbols
from gstlal import paths as gstlal_config_paths


__all__ = [
	"instruments_rate_given_noise",
	"P_instruments_given_signal",
	"SNRPDF",
	"NumeratorSNRCHIPDF"
]


#
# =============================================================================
#
#                              Instrument Combos
#
# =============================================================================
#


def P_instruments_given_signal(horizon_distances, n_samples = 500000, min_instruments = 2, min_distance = 0.):
	"""
	Example:

	>>> P_instruments_given_signal({"H1": 120., "L1": 120.})
	{frozenset(['H1', 'L1']): 1.0}
	>>> P_instruments_given_signal({"H1": 120., "L1": 120.}, min_instruments = 1)
	{frozenset(['L1']): 0.25423904879460091, frozenset(['H1', 'L1']): 0.49116512120682387, frozenset(['H1']): 0.25459582999857527}
	"""
	if n_samples < 1:
		raise ValueError("n_samples=%d must be >= 1" % n_samples)
	if min_distance < 0.:
		raise ValueError("min_distance=%g must be >= 0" % min_distance)
	if min_instruments < 1:
		raise ValueError("min_instruments=%d must be >= 1" % min_instruments)

	# get instrument names
	names = tuple(horizon_distances.keys())
	# get the horizon distances in that same order
	DH = numpy.array(tuple(horizon_distances.values()))
	# get detecor responses in that same order
	resps = [lalsimulation.DetectorPrefixToLALDetector(str(inst)).response for inst in names]

	# initialize output.  dictionary mapping instrument combination to
	# probability (initially all 0).
	result = dict.fromkeys((frozenset(instruments) for n in range(min_instruments, len(names) + 1) for instruments in itertools.combinations(names, n)), 0.0)
	if not result:
		raise ValueError("not enough instruments in horizon_distances to satisfy min_instruments")

	# check for no-op
	if (DH != 0.).sum() < min_instruments:
		# not enough instruments are on to form a coinc with the
		# minimum required instruments.  this is not considered an
		# error condition, return all probabilities = 0.  NOTE:
		# result is not normalizable.
		return result

	# we select random uniformly-distributed right ascensions so
	# there's no point in also choosing random GMSTs and any value is
	# as good as any other
	gmst = 0.0

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
			# instruments can see sources in this configuration
			# far enough way in this direction to form coincs.
			# if this happens too often we report a convergence
			# rate failure
			# FIXME:  min_distance is a misnomer, it's compared
			# to a volume-like thing with a mystery
			# proportionality constant factored out, so who
			# knows what to call it.  who cares.
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
	# make sure it's normalized.  allow an all-0 result in the event
	# that too few instruments are available to ever form coincs
	#

	total = sum(sorted(result.values()))
	assert abs(total - 1.) < 1e-13 or total == 0., "result insufficiently well normalized: %s, sum = %g" % (result, total)
	if total != 0.:
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
	# cache of pre-computed P(instruments | horizon distances, signal)
	# probabilities and P(SNRs | instruments, horizon distances,
	# signal) PDFs.
	#

	DEFAULT_FILENAME = os.path.join(gstlal_config_paths["pkgdatadir"], "inspiral_snr_pdf.xml.gz")
	snr_joint_pdf_cache = {}

	@ligolw_array.use_in
	@ligolw_param.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass


	class cacheentry(object):
		"""
		One entry in the SNR PDF cache.  For internal use only.
		"""
		def __init__(self, lnP_instruments, pdf, binnedarray):
			self.lnP_instruments = lnP_instruments
			self.pdf = pdf
			self.binnedarray = binnedarray


	# FIXME:  is the default choice of distance quantization appropriate?
	def __init__(self, snr_cutoff, log_distance_tolerance = math.log(1.05), min_ratio = 0.1):
		"""
		snr_cutoff sets the minimum SNR below which it is
		impossible to obtain a candidate (the trigger SNR
		threshold).

		min_ratio sets the minimum sensitivity an instrument must
		achieve to be considered to be on at all.  An instrument
		whose horizon distance is less than min_ratio * the horizon
		distance of the most sensitive instrument is considered to
		be off (it's horizon distance is set to 0).
		"""
		if log_distance_tolerance <= 0.:
			raise ValueError("require log_distance_tolerance > 0")
		if not 0. <= min_ratio < 1.:
			raise ValueError("require 0 <= min_ratio < 1")
		self.snr_cutoff = snr_cutoff
		self.log_distance_tolerance = log_distance_tolerance
		self.min_ratio = min_ratio


	def quantize_horizon_distances(self, horizon_distances):
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
		min_distance = self.min_ratio * horizon_distance_norm
		return dict((instrument, (NegInf if horizon_distance < min_distance else int(round(math.log(horizon_distance / horizon_distance_norm) / self.log_distance_tolerance)))) for instrument, horizon_distance in horizon_distances.items())


	@property
	def quants(self):
		return [NegInf] + range(int(math.ceil(math.log(self.min_ratio) / self.log_distance_tolerance)), 1)


	def quantized_horizon_distances(self, quants):
		if math.isinf(self.log_distance_tolerance):
			return dict((instrument, 0. if math.isinf(quant) else 1.) for instrument, quant in quants)
		return dict((instrument, math.exp(quant * self.log_distance_tolerance)) for instrument, quant in quants)


	def snr_joint_pdf_keyfunc(self, instruments, horizon_distances, min_instruments):
		"""
		Internal function defining the key for cache:  two element
		tuple, first element is frozen set of instruments for which
		this is the PDF, second element is frozen set of
		(instrument, horizon distance) pairs for all instruments in
		the network.  horizon distances are normalized to fractions
		of the largest among them and then the fractions aquantized
		to integer powers of a common factor
		"""
		return frozenset(instruments), frozenset(self.quantize_horizon_distances(horizon_distances).items()), min_instruments


	def get_snr_joint_pdf_binnedarray(self, instruments, horizon_distances, min_instruments):
		return self.snr_joint_pdf_cache[self.snr_joint_pdf_keyfunc(instruments, horizon_distances, min_instruments)].binnedarray


	def get_snr_joint_pdf(self, instruments, horizon_distances, min_instruments):
		return self.snr_joint_pdf_cache[self.snr_joint_pdf_keyfunc(instruments, horizon_distances, min_instruments)].pdf


	def lnP_instruments(self, instruments, horizon_distances, min_instruments):
		return self.snr_joint_pdf_cache[self.snr_joint_pdf_keyfunc(instruments, horizon_distances, min_instruments)].lnP_instruments


	def lnP_snrs(self, snrs, horizon_distances, min_instruments):
		"""
		snrs is an instrument-->SNR mapping containing one entry
		for each instrument that sees a signal.  snrs cannot
		contain instruments that are not listed in
		horizon_distances.

		horizon_distances is an instrument-->horizon distance mapping
		containing one entry for each instrument in the network.
		For instruments that are off set horizon distance to 0.
		"""
		# retrieve the PDF using the keys of the snrs mapping for
		# the participating instruments
		lnpdf = self.get_snr_joint_pdf(snrs, horizon_distances, min_instruments)
		# PDF axes are in alphabetical order
		return lnpdf(*(snr for instrument, snr in sorted(snrs.items())))


	def add_to_cache(self, horizon_distances, min_instruments, verbose = False):
		#
		# input check
		#

		if len(horizon_distances) < min_instruments:
			raise ValueError("require at least %d instruments, got %s" % (min_instruments, ", ".join(sorted(horizon_distances))))

		#
		# compute P(instruments | horizon distances, signal)
		#

		if verbose:
			print >>sys.stderr, "For horizon distances %s:" % ", ".join("%s = %.4g Mpc" % item for item in sorted(horizon_distances.items()))

		P_instruments = P_instruments_given_signal(
			self.quantized_horizon_distances(self.quantize_horizon_distances(horizon_distances).items()),
			min_instruments = min_instruments,
			n_samples = 1000000
		)

		if verbose:
			for key_value in sorted((",".join(sorted(key)), value) for key, value in P_instruments.items()):
				print >>sys.stderr, "\tP(%s | signal) = %g" % key_value
			print >>sys.stderr, "generating P(snrs | signal) ..."

		#
		# compute P(snrs | instruments, horizon distances, signal)
		#

		for n in range(min_instruments, len(horizon_distances) + 1):
			for instruments in itertools.combinations(sorted(horizon_distances), n):
				instruments = frozenset(instruments)
				key = self.snr_joint_pdf_keyfunc(instruments, horizon_distances, min_instruments)
				# already in cache?
				if key in self.snr_joint_pdf_cache:
					continue
				# need to build
				with ProgressBar(text = "%s candidates" % ", ".join(sorted(instruments))) if verbose else None as progressbar:
					binnedarray = self.joint_pdf_of_snrs(instruments, self.quantized_horizon_distances(key[1]), self.snr_cutoff, progressbar = progressbar)
				lnbinnedarray = binnedarray.copy()
				with numpy.errstate(divide = "ignore"):
					lnbinnedarray.array = numpy.log(lnbinnedarray.array)
				pdf = rate.InterpBinnedArray(lnbinnedarray, fill_value = NegInf)
				self.snr_joint_pdf_cache[key] = self.cacheentry(math.log(P_instruments[instruments]), pdf, binnedarray)


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
		if progressbar is not None:
			progressbar.max = n_samples

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
			if progressbar is not None:
				progressbar.update(progressbar.max)
			return pdf

		# we select random uniformly-distributed right ascensions
		# so there's no point in also choosing random GMSTs and any
		# value is as good as any other
		gmst = 0.0

		# run the sampler the requested # of iterations.  save some
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
			if progressbar is not None:
				progressbar.increment()
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
		self = cls(
			snr_cutoff = ligolw_param.get_pyvalue(xml, u"snr_cutoff"),
			log_distance_tolerance = ligolw_param.get_pyvalue(xml, u"log_distance_tolerance"),
			min_ratio = ligolw_param.get_pyvalue(xml, u"min_ratio")
		)
		for elem in xml.childNodes:
			if elem.tagName != ligolw.LIGO_LW.tagName:
				continue
			binnedarray = rate.BinnedArray.from_xml(elem, elem.Name.rsplit(u":", 1)[0])

			key = (
				frozenset(lsctables.instrumentsproperty.get(ligolw_param.get_pyvalue(elem, u"instruments:key"))),
				frozenset((inst.strip(), float(quant) if math.isinf(float(quant)) else int(quant)) for inst, quant in (inst_quant.split(u"=") for inst_quant in ligolw_param.get_pyvalue(elem, u"quantizedhorizons:key").split(u","))),
				ligolw_param.get_pyvalue(elem, u"min_instruments:key")
			)

			lnbinnedarray = binnedarray.copy()
			with numpy.errstate(divide = "ignore"):
				lnbinnedarray.array = numpy.log(lnbinnedarray.array)
			self.snr_joint_pdf_cache[key] = self.cacheentry(ligolw_param.get_pyvalue(elem, u"lnp_instruments"), rate.InterpBinnedArray(lnbinnedarray, fill_value = NegInf), binnedarray)
		return self


	def to_xml(self, name = u"generic"):
		xml = ligolw.LIGO_LW()
		xml.Name = u"%s:%s" % (name, u"inspiral_snr_pdf")
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"snr_cutoff", self.snr_cutoff))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"log_distance_tolerance", self.log_distance_tolerance))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"min_ratio", self.min_ratio))
		for i, (key, entry) in enumerate(self.snr_joint_pdf_cache.items()):
			elem = xml.appendChild(entry.binnedarray.to_xml(u"%d:pdf" % i))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"lnp_instruments", entry.lnP_instruments))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"instruments:key", lsctables.instrumentsproperty.set(key[0])))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"quantizedhorizons:key", u",".join(u"%s=%.17g" % inst_quant for inst_quant in sorted(key[1]))))
			elem.appendChild(ligolw_param.Param.from_pyvalue(u"min_instruments:key", key[2]))
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


def lnP_dt_dphi_uniform(coincidence_window_extension):
	# NOTE Currently hardcoded for H1L1
	# NOTE this is future proofed so that in > 2 ifo scenarios, the
	# appropriate length can be chosen for the uniform dt distribution
	return lnP_dt_dphi_uniform_H1L1(coincidence_window_extension)


def lnP_dt_dphi_signal(snrs, phase, dt, horizons, coincidence_window_extension):
	# Return P(dt, dphi|{rho_{IFO}}, signal)
	# FIXME Insert actual signal models
	if sorted(dt.keys()) == ("H1", "L1"):
		delta_t = float(dt["H1"] - dt["L1"])
		delta_phi = (phase["H1"] - phase["L1"]) % (2*math.pi)
		combined_snr = math.sqrt(snrs["H1"]**2. + snrs["L1"]**2.)
		if horizons["H1"] != 0 and horizons["L1"] != 0:
			# neither are zero, proceed as normal
			H1_snr_over_horizon = snrs["H1"] / horizons["H1"]
			L1_snr_over_horizon = snrs["L1"] / horizons["L1"]

		elif horizons["H1"] == horizons["L1"]:
			# both are zero, treat as equal
			H1_snr_over_horizon = snrs["H1"]
			L1_snr_over_horizon = snrs["L1"]

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
		return lnP_dt_dphi_uniform(coincidence_window_extension)


#
# =============================================================================
#
#                               SNR, \chi^2 PDF
#
# =============================================================================
#


class NumeratorSNRCHIPDF(rate.BinnedLnPDF):
	"""
	Reports ln P(chi^2/rho^2 | rho, signal)
	"""
	# NOTE:  by happy coincidence numpy's broadcasting rules allow the
	# inherited .at_centres() method to be used as-is
	def __init__(self, *args, **kwargs):
		super(rate.BinnedLnPDF, self).__init__(*args, **kwargs)
		self.volume = self.bins[1].upper() - self.bins[1].lower()
		# FIXME: instead of .shape and repeat() use .broadcast_to()
		# when we can rely on numpy >= 1.8
		#self.volume = numpy.broadcast_to(self.volume, self.bins.shape)
		self.volume.shape = (1, len(self.volume))
		self.volume = numpy.repeat(self.volume, len(self.bins[0]), 0)
		self.norm = numpy.zeros((len(self.bins[0]), 1))

	def __getitem__(self, coords):
		return numpy.log(super(rate.BinnedLnPDF, self).__getitem__(coords)) - self.norm[self.bins(*coords)[0]]

	def marginalize(self, *args, **kwargs):
		raise NotImplementedError

	def __iadd__(self, other):
		super(rate.BinnedLnPDF, self).__iadd__(other)
		self.norm = numpy.log(numpy.exp(self.norm) + numpy.exp(other.norm))
		return self

	def normalize(self):
		# replace the vector with a new one so that we don't
		# interfere with any copies that might have been made
		with numpy.errstate(divide = "ignore"):
			self.norm = numpy.log(self.array.sum(axis = 1))
		self.norm.shape = (len(self.norm), 1)

	@staticmethod
	def add_signal_model(lnpdf, n, prefactors_range, df, inv_snr_pow = 4., snr_min = 4., progressbar = None):
		if df <= 0.:
			raise ValueError("require df >= 0: %s" % repr(df))
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 100)
		if progressbar is not None:
			progressbar.max = len(pfs)

		# FIXME:  except for the low-SNR cut, the slicing is done
		# to work around various overflow and loss-of-precision
		# issues in the extreme parts of the domain of definition.
		# it would be nice to identify the causes of these and
		# either fix them or ignore them one-by-one with a comment
		# explaining why it's OK to ignore the ones being ignored.
		# for example, computing snrchi2 by exponentiating the sum
		# of the logs of the terms might permit its evaluation
		# everywhere on the domain.  can ncx2pdf() be made to work
		# everywhere?
		snrindices, rcossindices = lnpdf.bins[snr_min:1e10, 1e-10:1e10]
		snr, dsnr = lnpdf.bins[0].centres()[snrindices], lnpdf.bins[0].upper()[snrindices] - lnpdf.bins[0].lower()[snrindices]
		rcoss, drcoss = lnpdf.bins[1].centres()[rcossindices], lnpdf.bins[1].upper()[rcossindices] - lnpdf.bins[1].lower()[rcossindices]

		snr2 = snr**2.
		snrchi2 = numpy.outer(snr2, rcoss) * df

		arr = numpy.zeros_like(lnpdf.array)
		for pf in pfs:
			if progressbar is not None:
				progressbar.increment()
			arr[snrindices, rcossindices] += gstlalstats.ncx2pdf(snrchi2, df, numpy.array([pf * snr2]).T)

		# convert to counts by multiplying by bin volume, and also
		# multiply by an SNR powr law
		arr[snrindices, rcossindices] *= numpy.outer(dsnr / snr**inv_snr_pow, drcoss)

		# normalize to a total count of n
		arr *= n / arr.sum()

		# add to lnpdf
		lnpdf.array += arr

	def to_xml(self, *args, **kwargs):
		elem = super(rate.BinnedLnPDF, self).to_xml(*args, **kwargs)
		elem.appendChild(ligolw_array.Array.build("norm", self.norm))
		return elem

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(rate.BinnedLnPDF, cls).from_xml(xml, name)
		self.norm = ligolw_array.get_array(xml, "norm").array
		return self


#
# =============================================================================
#
#                 Utilities for computing SNR, time, phase PDFs
#
# =============================================================================
#


class HyperRect(object):
	__slots__ = ["lower", "upper", "parent", "left", "right", "point", "prob", "nodeid"]
	def __init__(self, lower, upper, parent, point = None, prob = None, nodeid = None):
		"""
		An N dimensional hyper rectangle with references to sub
		rectangles split along one dimension so that a binary tree of such objects can
		be constructed.  NOTE for now each dimension must have no more than 128 bins in
		any given side.  FIXME, make this an option to allow larger histograms by using
		e.g., uint16.
		"""
		self.lower = numpy.array(lower, dtype="uint8")
		self.upper = numpy.array(upper, dtype="uint8")
		self.parent = parent
		self.left = None
		self.right = None
		self.point = point
		self.prob = prob
		self.nodeid = nodeid

	def serialize(self, prob):
		"""
		Return the minimal set of information to describe a
		hyperrectangle in "read only" mode.
		"""
		return [self.lower, self.upper, self.parent, self.left, self.right, prob]

	def split(self, splitdim):
		"""
		return two new hyper rectangles split along splitdim and save
		references to them in this instance.
		"""
		upper_left = self.upper.copy()
		upper_right = self.upper.copy()
		lower_left = self.lower.copy()
		lower_right = self.lower.copy()
		upper_left[splitdim] -= (self.upper[splitdim] - self.lower[splitdim])/ 2.
		lower_right[splitdim] += (self.upper[splitdim] - self.lower[splitdim])/ 2.
		self.left = HyperRect(lower_left, upper_left, self)
		self.right = HyperRect(lower_right, upper_right, self)

		return self.left, self.right

	def __contains__(self, point):
		"""
		Determine if a point is contained within this hyper rectangle
		"""
		return (numpy.logical_and(point >= self.lower, point < self.upper)).all()

	def biggest_side(self):
		"""
		Compute the biggest side. This is used in the splitting process
		to ensure the most square tiles possible.  NOTE, in the future perhaps we
		should allow the user to override this with a different function.
		"""
		return numpy.argmax(self.upper - self.lower)

	@property
	def size(self):
		return self.upper - self.lower

	@classmethod
	def search(cls, instance, point):
		"""
		Recursively descend starting with the rectangle defined by
		instance until a rectangle with no right or left siblings is found.
		"""
		if instance.left is None and instance.right is None:
			return instance
		if point in instance.left:
			return HyperRect.search(instance.left, point)
		elif point in instance.right:
			return HyperRect.search(instance.right, point)
		else:
			raise ValueError("This should be impossible: %s", point)

	def __repr__(self):
		return "lower: %s, upper: %s" % (repr(self.lower), repr(self.upper))


class DynamicBins(object):
	def __init__(self, bin_edges, static = False):
		"""
		A class for turning a bunch of hyper rectangles into a dynamic histogram

		Bins should be a tuple of arrays where the length of the
		tuple is the dimension of the function space that you are
		histogramming and each array in the tuple is the explicit bin
		boundaries in the ith dimension.  The length of the arrays
		should always be expressable as 2^x +1, i.e., there will be a
		power of two number of bins described by that number of bins
		+1 bin edges.  For the time being a max of 128 bins are allowed,
		thus the array length must be <= 129.  This could be changed
		at a later time.  To be explicit, the bin edge lengths must be
		one of 3, 5, 9, 17, 33, 65, 129.

		Supports a "read only" mode if static is True.  Currently this
		is only done through reading an instance off disk with hdf5.
		"""

		self.max_splits = numpy.array([len(b) - 1 for b in bin_edges])
		assert numpy.all(self.max_splits <= 128)
		assert not numpy.any(self.max_splits % 2)
		self.bin_edges = tuple([numpy.array(b, dtype=float) for b in bin_edges])
		#self.__finish_init()

		# We assume that before we start there is a probability of 1 of being in the full cell
		if not static:
			self.num_points = 0
			self.total_prob = 1.0
			self.num_cells = 1
			self.hyper_rect = HyperRect(self.point_to_bin([b[0] for b in self.bin_edges]), self.point_to_bin([b[-1] for b in self.bin_edges]), None)
			self.hyper_rect.prob = 1.0
			self.static_data = None
		else:
			self.hyper_rect = None

	def __repr__(self):
		out = "\nThis is a %dD dynamic histogram\n" % len(self.bin_edges)
		out += "\tMaximum bins in each dimension: %s\n" % self.max_splits
		out += "\tLower boundary: %s\n" % [b[0] for b in self.bin_edges]
		out += "\tUpper boundary: %s\n" % [b[-1] for b in self.bin_edges]
		out += "\tCurrently contains %d points in %d cells\n" % (self.num_points, self.num_cells)
		out += "\tCurrently contains %f probability\n" % (self.total_prob)
		return out

	@staticmethod
	def _from_hdf5(f):
		"""
		Instantiate the class from an open HDF5 file handle, f, which
		is rooted at the path where it needs to be to get the information below.
		"""
		dgrp = f["dynbins"]
		begrp = dgrp["bin_edges"]
		DB = DynamicBins([numpy.array(begrp["%d" % i]) for i in range(len(begrp.keys()))], static = True)
		DB.num_points = dgrp.attrs["num_points"]
		DB.total_prob = dgrp.attrs["total_prob"]
		DB.num_cells = dgrp.attrs["num_cells"]

		bgrp = f["bins"]
		lower = numpy.array(bgrp["lower"])
		upper = numpy.array(bgrp["upper"])
		parent = numpy.array(bgrp["parent"])
		right = numpy.array(bgrp["right"])
		left = numpy.array(bgrp["left"])
		prob = numpy.array(bgrp["prob"])
		# This order is assumed elsewhere (lower, upper, parent, right, left, prob)
		DB.static_data = (lower, upper, parent, right, left, prob)
		return DB

	@staticmethod
	def from_hdf5(fname):
		"""
		Instantiate the class from an hdf5 file name that is not yet open
		"""
		f = h5py.File(fname, "r")
		DB = self._from_hdf5(f)
		f.close()
		return DB

	def point_to_bin(self, point):
		"""
		Figure out which bin a given point belongs in
		"""
		# search sorted right gives the index above the value in
		# question, so remove 1, this is how we get it to be [)
		point = numpy.array([a.searchsorted(point[i], "right")-1 for i,a in enumerate(self.bin_edges)])
		assert numpy.all(point <= 128) and numpy.all(point >=0)
		point = numpy.array(point, dtype = "uint8")
		return point	

	def bin_to_point(self, binpoint):
		"""
		Return a point which is the lower corner of a given bin.
		"""
		return numpy.array([self.bin_edges[i][p] for i,p in enumerate(binpoint)])

	def insert(self, point, prob):
		"""
		Add a new point to the histogram.

		There are three possibilities
			1) The hyperrectangle is empty
			2) It is not empty, but is as small as it is allowed to be
			3) It is not empty and is larger than the minimum size, 
			so it will be split
		"""
		assert self.static_data is None and self.hyper_rect is not None

		# handle a special case of a 0D histogram
		if len(self.bin_edges) == 0:
			self.num_points += 1
			self.total_prob += prob
			return

		# figure out the hyperrectangle where this point belongs
		binpoint = self.point_to_bin(point)
		thisrect = HyperRect.search(self.hyper_rect, binpoint)


		# Case 1)
		if thisrect.point is None:
			assert thisrect.prob is not None
			thisrect.point = binpoint
			thisrect.prob += prob
			self.num_points += 1
			self.total_prob += prob
		# Case 2) or 3)
		else:
			# Case 2)
			# Check to see if we have reached the maximum resolution first
			splitdim = thisrect.biggest_side()
			if thisrect.size[splitdim] / 2. < 1:
				thisrect.prob += prob
				self.num_points += 1
				self.total_prob += prob
				assert not thisrect.left and not thisrect.right
			# Case 3)
			else:
				assert thisrect.point is not None
				left, right = thisrect.split(splitdim)
				self.num_points += 1
				self.num_cells += 1
				self.total_prob += prob
				if binpoint in left:
					left.point = binpoint
				elif binpoint in right:
					right.point = binpoint
				else:
					raise ValueError("This is impossible: %s" % binpoint)
				right.prob = (prob + thisrect.prob) / 2.
				left.prob = (prob + thisrect.prob) / 2.
				thisrect.point = None
				thisrect.prob = None

	def __call__(self, point):
		"""
		Return the value of the PDF at point.  If called over the
		actual histogram cells, it should integrate to 1. For example, say you have an
		instance of this class called DB:

		>>> totalprobvol = 0.
		>>> for (lower, upper, volume, prob) in DB:
		>>>	totalprobvol += prob * volume
		>>> print totalprobvol
		>>> 1.0
		"""
		binpoint = self.point_to_bin(point)
		if self.static_data is None:
			return self.__eval_prob(binpoint)
		else:
			return self.__static_prob(binpoint)

	def __eval_prob(self, binpoint):
		node = HyperRect.search(self.hyper_rect, binpoint)
		l = numpy.array([self.bin_edges[i][p] for i,p in enumerate(node.lower)])
		u = numpy.array([self.bin_edges[i][p] for i,p in enumerate(node.upper)])
		volume = numpy.product(u - l)
		return node.prob / self.total_prob / volume

	def __static_prob(self, binpoint):

		def _in_node(point, upper, lower):
			return (numpy.logical_and(point >= lower, point < upper)).all()

		def search_serialized(staticbins, point, ix):
			lower, upper, parent, right, left, prob = staticbins
			# we have found our terminal cell
			leftix = left[ix]
			rightix = right[ix]
			if leftix == 0 and rightix == 0:
				return prob[ix]
			# check to see if it is in the left
			if _in_node(point, upper[leftix,:], lower[leftix,:]):
				return search_serialized(staticbins, point, leftix)
			# check to see if it is in the right
			elif _in_node(point, upper[rightix,:], lower[rightix,:]):
				return search_serialized(staticbins, point, rightix)
			else:
				raise ValueError("This should be impossible: %s" % point)

		# This order is assumed elsewhere (lower, upper, parent, right, left, prob)
		return search_serialized(self.static_data, binpoint, ix = 1)

	def __serialize(self):
		assert self.static_data is None
		out = []
		nodes = self.__flatten()
		# setup efficient storage for all of the data
		# for an 8D PDF this should ad up to 32 bytes per bin
		# NOTE we will use zero to denote a null value so we want to start counting at 1
		# FIXME should probably check that we don't have more than 4 billion bins.  
		LOWER = numpy.zeros((len(nodes)+1, len(self.bin_edges)), dtype = "uint8")
		UPPER = numpy.zeros((len(nodes)+1, len(self.bin_edges)), dtype = "uint8")
		PARENT = numpy.zeros(len(nodes)+1, dtype="uint32")
		LEFT = numpy.zeros(len(nodes)+1, dtype="uint32")
		RIGHT = numpy.zeros(len(nodes)+1, dtype="uint32")
		PROB = numpy.zeros(len(nodes)+1, dtype="float32")
		
		# give them all an integer id first
		# NOTE we will use zero to denote a null value so we want to start counting at 1
		for i,node in enumerate(nodes):
			node.nodeid = i+1
		for i,node in enumerate(nodes):
			LOWER[i+1,:] = node.lower
			UPPER[i+1,:] = node.upper
			PARENT[i+1] = node.parent.nodeid if node.parent else 0
			RIGHT[i+1] = node.right.nodeid if node.right else 0
			LEFT[i+1] = node.left.nodeid if node.left else 0
			PROB[i+1] = self.__eval_prob(node.lower) # this gives the probability of this cell
		return LOWER, UPPER, PARENT, RIGHT, LEFT, PROB

	def __flatten(self, this_rect = None, out = None, leaf = False):
		# FIXME, can this be a generator??
		assert self.static_data is None
		if out is None:
			out = []
		if this_rect == None:
			this_rect = self.hyper_rect
		if not leaf:
			out.append(this_rect)
		if leaf and not this_rect.right and not this_rect.left:
			out.append(this_rect)
		if this_rect.right:
			self.__flatten(this_rect.right, out, leaf)
		if this_rect.left:
			self.__flatten(this_rect.left, out, leaf)

		return out

	def bins(self):
		"""
		Iterate over the histogram bins
		"""
		nodes = self.__flatten(leaf = True)
		for node in nodes:
			lower = numpy.array([self.bin_edges[i][p] for i,p in enumerate(node.lower)])
			upper = numpy.array([self.bin_edges[i][p] for i,p in enumerate(node.upper)])
			volume = numpy.product(upper - lower)
			prob = node.prob / self.total_prob / volume
			yield self.bin_to_point(node.lower), self.bin_to_point(node.upper), volume, prob

	def sbins(self):
		"""
		Iterate over the histogram bins in read only mode
		"""
		# NOTE the first element of static bins is reserved as a NULL value, it does not contain a valid bin
		lower = self.static_data[0]
		upper = self.static_data[1]
		prob = self.static_data[5]
		right = self.static_data[3]
		left = self.static_data[4]
		for i, p in enumerate(prob):
			if i == 0 or right[i] != 0 or left[i] != 0:
				continue
			l = numpy.array([self.bin_edges[j][x] for j,x in enumerate(lower[i,:])])
			u = numpy.array([self.bin_edges[j][x] for j,x in enumerate(upper[i,:])])
			v = numpy.product(u - l)
			yield self.bin_to_point(lower[i,:]), self.bin_to_point(upper[i,:]), v, p

	def __iter__(self):
		if self.static_data is None:
			return self.bins()
		else:
			return self.sbins()

	def to_hdf5(self, fname):
		assert self.static_data is None
		f = h5py.File(fname, "w")
		self._to_hdf5(f)
		f.close()

	def _to_hdf5(self, f):
		dgrp = f.create_group("dynbins")
		begroup = dgrp.create_group("bin_edges")
		for i, bin_edge in enumerate(self.bin_edges):
			begroup.create_dataset("%d" % i, data = bin_edge)
		dgrp.attrs["num_points"] = self.num_points 
		dgrp.attrs["total_prob"] = self.total_prob
		dgrp.attrs["num_cells"] = self.num_cells

		bgrp = f.create_group("bins")
		lower, upper, parent, right, left, prob = self.__serialize()
		bgrp.create_dataset("lower", data = lower)
		bgrp.create_dataset("upper", data = upper)
		bgrp.create_dataset("parent", data = parent)
		bgrp.create_dataset("right", data = right)
		bgrp.create_dataset("left", data = left)
		bgrp.create_dataset("prob", data = prob)


#
# Helper class to do lal FFTs as a drop in replacement for scipy
#

class FFT(object):
	def __init__(self):
		self.fwdplan = {}
		self.tvec = {}
		self.fvec = {}

	def __call__(self, arr):
		length = len(arr)
		if length not in self.fwdplan:
			self.fwdplan[length] = lal.CreateForwardREAL4FFTPlan(length, 1)
		if length not in self.tvec:
			self.tvec[length] = lal.CreateREAL4Vector(length)
		if length not in self.fvec:
			self.fvec[length] = lal.CreateCOMPLEX8Vector(length / 2 + 1)

		self.tvec[length].data[:] = arr[:]
		lal.REAL4ForwardFFT(self.fvec[length], self.tvec[length], self.fwdplan[length])

		return numpy.array(self.fvec[length].data, dtype=complex)

class IFFT(object):
	def __init__(self):
		self.revplan = {}
		self.tvec = {}
		self.fvec = {}

	def __call__(self, arr):
		length = 2 * (len(arr) -1)
		if length not in self.revplan:
			self.revplan[length] = lal.CreateReverseREAL4FFTPlan(length, 1)
		if length not in self.tvec:
			self.tvec[length] = lal.CreateREAL4Vector(length)
		if length not in self.fvec:
			self.fvec[length] = lal.CreateCOMPLEX8Vector(length / 2 + 1)

		self.fvec[length].data[:] = arr[:] / length
		lal.REAL4ReverseFFT(self.tvec[length], self.fvec[length], self.revplan[length])

		return numpy.array(self.tvec[length].data, dtype=float)

class RandomSource(object):

	def __init__(self, psd, horizons, SR = 2048, FL = 32, FH = 1024, DF = 32):

		psd_data = psd.data.data
		f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF

		f = f[int(FL / psd.deltaF):int(FH / psd.deltaF)]
		psd = psd_data[int(FL / psd.deltaF): int(FH / psd.deltaF)]

		psd = interpolate.interp1d(f, psd)
		f = numpy.arange(FL, FH, DF)
		PSD = psd(f)

		self.t = numpy.arange(0, SR) / float(SR) / DF
		self.f = numpy.arange(-SR/2, SR/2, DF)
		psd = numpy.ones(len(self.f)) * 1e-30
		psd[(SR/2-FH)/DF:(SR/2-FL)/DF] = PSD[::-1]
		psd[(SR/2+FL)/DF:(SR/2+FH)/DF] = PSD

		self.psd = psd

		self.horizons = horizons

		self.responses = {"H1": lal.CachedDetectors[lal.LHO_4K_DETECTOR].response, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].response, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].response}
		self.locations = {"H1":lal.CachedDetectors[lal.LHO_4K_DETECTOR].location, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].location, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].location}

		self.fft = FFT()
		self.ifft = IFFT()

		self.w1t = self.waveform(0., 0.)
		self.w1 = self.fft(self.w1t)
		self.w2 = self.fft(self.waveform(0., numpy.pi / 2.))

	def __call__(self):
		# We will draw uniformly in extrinsic parameters
		# FIXME Coalescence phase?

		COSIOTA = numpy.random.uniform(-1.0,1.0)
		RA = numpy.random.uniform(0.0,2.0*numpy.pi)
		DEC = numpy.arcsin(numpy.random.uniform(-1.0,1.0))
		PSI = numpy.random.uniform(0.0,2.0*numpy.pi)
		# FIXME assumes an SNR 4 threshold
		#D = numpy.random.power(3) * max(self.horizons.values()) * 2
		D = numpy.random.power(1) * max(self.horizons.values()) * 2
		PROB = D**2
		T = lal.LIGOTimeGPS(0)

		# Derived quantities
		GMST = lal.GreenwichMeanSiderealTime(T)
		hplus = 0.5 * (1.0 + COSIOTA**2)
		hcross = COSIOTA

		phi = {}
		Deff = {}
		time = {}
		snr = {}
		for ifo in self.horizons:

			#
			# These are the fiducial values
			#

			Fplus, Fcross = lal.ComputeDetAMResponse(self.responses[ifo], RA, DEC, PSI, GMST)
			phi[ifo] = - numpy.arctan2(Fcross * hcross, Fplus * hplus)
			Deff[ifo] = D / ((Fplus * hplus)**2 + (Fcross * hcross)**2)**0.5
			time[ifo] = lal.TimeDelayFromEarthCenter(self.locations[ifo], RA, DEC, T)
			snr[ifo] = self.horizons[ifo] / Deff[ifo] * 8.

		#
		# Now we will slide stuff around according to the
		# uncertainty in matched filtering NOTE.  We target a
		# network SNR of 11.  The actual binning code will use
		# SNR ratios, so we care most about capturing the
		# uncertainty right when things get interesting.
		#

		network_snr = (numpy.array(snr.values())**2).sum()**.5
		adjustment_factor = 11. / network_snr
		for ifo in self.horizons:
			newsnr, dt, dphi = self.sample_from_matched_filter(snr[ifo] * adjustment_factor)
			snr[ifo] += (newsnr - snr[ifo] * adjustment_factor)
			time[ifo] += dt
			phi[ifo] += dphi

		return time, phi, snr, Deff, PROB

	def match(self, w1, w2):
		return numpy.real((numpy.conj(w1) * w2).sum())

	def matchfft(self, w1, w2):
		return numpy.real(self.ifft(numpy.conj(w1) * w2))

	def waveform(self, tc, phi):
		a = abs(self.f)**(-7./6.)
		a[len(a)/2] = 0
		w = a * numpy.exp(numpy.pi * 2 * self.f * 1j * tc + phi * 1.j) / self.psd **.5
		w = numpy.real(fft(w))
		return numpy.array(w / self.match(w, w)**.5, dtype=float)

	def noise(self):
		return numpy.random.randn(len(self.f))

	def inject(self, w, n, A):
		return A*w+n

	def sample_from_matched_filter(self, snr):
		while True:
			n = self.noise()
			d = self.fft(self.inject(self.w1t, n, snr))
			m1 = self.matchfft(self.w1,d)
			m2 = self.matchfft(self.w2,d)
			snr = ((m1**2 + m2**2)**.5)
			ix = snr.argmax()
			# FIXME Only consider times within 3ms to be "found"
			if self.t[ix] > 0.003 and self.t[ix] < 0.997:
				continue
			if self.t[ix] >= 0.997:
				DT = (1. - self.t[ix])
			else:
				DT = self.t[ix]
			SNR = snr[ix]
			DPHI = numpy.arctan(m2[ix] / m1[ix])
			break
		return SNR, DT, DPHI

class InspiralExtrinsicParameters(object):

	def __init__(self, horizons, snr_thresh, min_instruments, histograms = None):
		self.horizons = horizons
		self.snr_thresh = snr_thresh
		self.min_instruments = min_instruments
		self.instruments = tuple(sorted(self.horizons.keys()))
		combos = set()
		for i in range(self.min_instruments, len(self.instruments) + 1):
			for choice in itertools.combinations(self.instruments, i):
				combos.add(tuple(sorted(choice)))
		self.combos = sorted(list(combos))
		if histograms is not None:
			self.histograms = histograms
		else:
			self.histograms = dict((combo, None) for combo in self.combos)
			for combo, bin_edges in self.boundaries().items():
				self.histograms[combo] = DynamicBins(bin_edges)

	@staticmethod
	def from_hdf5(fname):

		f = h5py.File(fname, "r")

		snr_thresh = f.attrs["snr_thresh"]
		min_instruments = f.attrs["min_instruments"]
		horizons = dict(f["horizons"].attrs)
		histograms = {}
		for combo in f["histograms"]:
			key = tuple(combo.split(","))
			histograms[key] = DynamicBins._from_hdf5(f["histograms"][combo])
		return InspiralExtrinsicParameters(horizons, snr_thresh, min_instruments, histograms)
		f.close()

	#def snrfunc(self, snr):
	#	return min(snr, self.max_snr)

	def effdratiofunc(self, effd1, effd2):
		return min(7.9999, max(0.125001, effd1 / effd2))

	def boundaries(self):
		out = {}
		def boundary(ifos):
			bin_edges = []
			ifos = sorted(ifos)
			# for now we only gaurantee that this will work for H1,
			# L1, V1, need to change time delay boundaries for
			# Kagra possibly
			assert not set(ifos) - set(["H1", "L1", "V1"])
			# we have pairs of dt and dphi
			for pair in list(itertools.combinations(ifos, 2))[:2]:
				# Assumes Virgo time delay is the biggest we care about
				bin_edges.append(numpy.linspace(-0.032, 0.032, 65))
			for pair in list(itertools.combinations(ifos, 2))[:2]:
				bin_edges.append(numpy.linspace(-numpy.pi, numpy.pi, 33))
			for pair in list(itertools.combinations(ifos, 2))[:2]:
				bin_edges.append(numpy.logspace(numpy.log10(1./8), numpy.log10(8), 17))
			# NOTE this would track the SNR instead of effective distance ratio
			#for ifo in ifos:
			#	bin_edges.append(numpy.logspace(numpy.log10(self.snr_thresh), numpy.log10(self.max_snr), 65))
			return bin_edges
		for combo in self.combos:
			out[combo] = boundary(combo)
		return out

	def pointfunc(self, time, phase, snr):
		ifos = sorted(time)
		assert 1 <= len(ifos) <= 3
		point = []
		# first time diff
		for pair in list(itertools.combinations(ifos, 2))[:2]:
			point.append(time[pair[0]] - time[pair[1]])
		# then phase diff
		for pair in list(itertools.combinations(ifos, 2))[:2]:
			unwrapped = numpy.unwrap([phase[pair[0]], phase[pair[1]]])
			point.append(unwrapped[0] - unwrapped[1])
		# NOTE this would track the SNR instead of effective distance ratio
		#for ifo in ifos:
		#	point.append(self.snrfunc(snr[ifo]))
		# then effective distance ratio
		for pair in list(itertools.combinations(ifos, 2))[:2]:
			point.append(self.effdratiofunc(self.horizons[pair[0]] / snr[pair[0]], self.horizons[pair[1]] / snr[pair[1]]))
		return point

	def insert(self, time, phase, snr, prob):
		detected_ifos = tuple(sorted(snr))
		self.histograms[detected_ifos].insert(self.pointfunc(time, phase, snr), prob)

	def __call__(self, time, phase, snr):
		ifos = tuple(sorted(time))
		if len(ifos) > 1:
			return self.histograms[ifos](self.pointfunc(time, phase, snr))
		else:
			return 1

	def to_hdf5(self, fname):
		f = h5py.File(fname, "w")
		# first record the minimal set of instance state
		f.attrs["snr_thresh"] = self.snr_thresh
		f.attrs["min_instruments"] = self.min_instruments
		#f.attrs["max_snr"] = self.max_snr

		# Then record the horizon distances
		hgrp = f.create_group("horizons")
		for ifo in self.horizons:
			hgrp.attrs[ifo] = self.horizons[ifo]

		# then all of the histogram data
		histgrp = f.create_group("histograms")
		for combo in self.combos:
			cgrp = histgrp.create_group(",".join(combo))
			self.histograms[combo]._to_hdf5(cgrp)

		f.close()