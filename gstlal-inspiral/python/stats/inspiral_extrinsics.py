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
import scipy
from scipy import stats
from scipy import spatial
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
from lalburst import snglcoinc
import lalsimulation


# FIXME:  caution, this information might get organized differently later.
# for now we just need to figure out where the gstlal-inspiral directory in
# share/ is.  don't write anything that assumes that this module will
# continue to define any of these symbols
from gstlal import paths as gstlal_config_paths


__all__ = [
	"InspiralExtrinsics",
	"TimePhaseSNR",
	"p_of_instruments_given_horizons",
	"margprob",
	"chunker",
]


__doc__ = """

The goal of this module is to implement the probability of getting a given set
of extrinsic parameters for a set of detectors parameterized by n-tuples of
trigger parameters: (snr, horizon distance, end time and phase) assuming that
the event is a gravitational wave signal, *s*, coming from an isotropic
distribution in location, orientation and the volume of space.  The
implementation of this in the calling code can be found in
:py:mod:`stats.inspiral_lr`.

The probabilities are factored in the following way:

.. math::

	P(\\vec{\\rho}, \\vec{t}, \\vec{\phi}, \\vec{O} | \\vec{D_H}, s)
	= 
	\underbrace{P(\\vec{\\rho}, \\vec{t}, \\vec{\phi} | \\vec{O}, \\vec{D_H}, s)}_{\mathrm{1:\,TimePhaseSNR()}} 
	\\times 
	\underbrace{P(\\vec{O} | \\vec{D_H}, s)}_{\mathrm{2:\,p\_of\_instruments\_given\_horizons()}}

where:

* :math:`\\vec{\\rho}` denotes the vector of SNRs with one component from each detector
* :math:`\\vec{t}`     denotes the vector of end time with one component from each detector
* :math:`\\vec{\phi}`  denotes the vector of measured phases with one component from each detector
* :math:`\\vec{O}`     denotes the vector of observing IFOs with one component from each detector.  Strictly speaking this is just a label that desribes what detectors the components of the other vectors correspond to
* :math:`\\vec{D_H}`   denotes the vector of horizon distances with one component from each detector
* :math:`s`            denotes the signal hypothesis

See the following for details:

* :py:class:`InspiralExtrinsics` -- helper class to implement the full probability expression
* :py:class:`TimePhaseSNR` -- implementation of term 1 above
* :py:class:`p_of_instruments_given_horizons` -- implementation of term 2 above

and :any:`gstlal_inspiral_plot_extrinsic_params` for some visualizations of
these PDFs.


Sanity Checks
-------------

The code here is new for O3.  We compared the result to the O2 code on 100,000
seconds of data searching for binary neutron stars in H and L.  The injection
set was identical.  Although an improvement for an HL search was not expected,
in fact it appears that the reimplementation is a bit more sensitive.

.. |O2_O3_O2_LR_range| image:: ../images/O2_O3_O2_LR_range.png
   :width: 400px

.. |O2_O3_O3_LR_range| image:: ../images/O2_O3_O3_LR_range.png
   :width: 400px

.. |O2_O3_O2_cnt_vs_LR| image:: ../images/O2_O3_O2_cnt_vs_LR.png
   :width: 400px

.. |O2_O3_O3_cnt_vs_LR| image:: ../images/O2_O3_O3_cnt_vs_LR.png
   :width: 400px

.. |O2_O3_LR_double_vs_triple| image:: ../images/O2_O3_LR_double_vs_triple.png
   :width: 400px

.. |O2_O3_HVtest| image:: ../images/HVtest.png
   :width: 400px

+-------------------+-------------------------+-------------------------+
|                   | O2 Code                 | O3 Code                 |
+===================+=========================+=========================+
| **Found/Missed**  | 939 / 2270              | 951 / 2258              |
+-------------------+-------------------------+-------------------------+
| **Range**         | |O2_O3_O2_LR_range|     | |O2_O3_O3_LR_range|     |
+-------------------+-------------------------+-------------------------+
| **Count vs LR**   | |O2_O3_O2_cnt_vs_LR|    | |O2_O3_O3_cnt_vs_LR|    |
+-------------------+-------------------------+-------------------------+

Double vs triple found injections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this check we ran two analyses:

- An HL only analysis
- An HLV analysis

Both were over the same time period.  A common set of found injections was
identified and the false alarm probability (FAP) was computed for the doubles
and triples. *Ideally* false alarm probabilities for all doubles would be
higher than all triples, but this is not a requirement since noise and
especially glitches can cause a triple to be ranked below a double.  The plot
below shows that at least the trend is correct.  NOTE we are not currently
computing doubles and triples and picking the best. 

|O2_O3_LR_double_vs_triple|

Check of PDFs
^^^^^^^^^^^^^

We tested the procedure for evaluating the probabilility using the procedure described below (orange) against a monte carlo (blue).  The details are in this script

https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/tests/dtdphitest.py

You can modify this source code to implement different checks.  The one
implemented is to plot the marginal distributions of time delay and phase delay
between Hanford and Virgo under the condition that the measured effective
distance is the same.  NOTE this test assumes the same covariance matrix for
noise as the code below in order to test the procedure, but it doesn't prove
that the assumption is optimal.

|O2_O3_HVtest|


Review Status
-------------

Do no harm check of O2 results  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Comparing runs before and after (done) 
- Checking the probabilities returned by new code and old code (sarah is working on it) to show consistent results

Check of error assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Calculate theoretical delta T, delta phi and snr ratio values of an O2 injection set.  Then compute same parameters from injections.  The difference between those (in e.g., a scatter plot) should give a sense of the errors on those parameters caused by noise.  (sarah is working on it)
- Eventually use the fisher matrix for the error estimates (chad will do it, but not for O2)


Inclusion of virgo
^^^^^^^^^^^^^^^^^^

- Virgo should not make the analysis worse in an average sense. (this has been demonstrated, but Chad will make it a bit more quantitative)
- Understand cases where / if virgo does downrank a trigger
- Consider having the likelihood calculation maximize over all trigger subsets (Chad and Kipp, but not for O2)


+-------------------------------------------------+------------------------------------------+------------+
| Names                                           | Hash                                     | Date       |
+=================================================+==========================================+============+
| --                                              | --                                       | --         |
+-------------------------------------------------+------------------------------------------+------------+


Documentation of classes and functions
--------------------------------------

"""


#
# =============================================================================
#
#                              Instrument Combos
#
# =============================================================================
#

# FIXME: This code is no longer used.
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


# FIXME: This code is no longer used.
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
			print >>sys.stderr, "For horizon distances %s, requiring %d instrument(s):" % (", ".join("%s = %.4g Mpc" % item for item in sorted(horizon_distances.items())), min_instruments)

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
				self.snr_joint_pdf_cache[key] = self.cacheentry(math.log(P_instruments[instruments]) if P_instruments[instruments] else NegInf, pdf, binnedarray)


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
#                               SNR, \chi^2 PDF
#
# =============================================================================
#


# FIXME: This code is no longer used.
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
		# the total count is meaningless, it serves only to set the
		# scale by which the density estimation kernel chooses its
		# size, so we preserve the count across this operation.  if
		# the two arguments have different counts, use the
		# geometric mean unless one of the two is 0 in which case
		# don't screw with the total count
		self_count, other_count = self.array.sum(), other.array.sum()
		super(rate.BinnedLnPDF, self).__iadd__(other)
		if self_count and other_count:
			self.array *= numpy.exp((numpy.log(self_count) + numpy.log(other_count)) / 2.) / self.array.sum()
		self.norm = numpy.log(numpy.exp(self.norm) + numpy.exp(other.norm))
		return self

	def normalize(self):
		# replace the vector with a new one so that we don't
		# interfere with any copies that might have been made
		with numpy.errstate(divide = "ignore"):
			self.norm = numpy.log(self.array.sum(axis = 1))
		self.norm.shape = (len(self.norm), 1)

	@staticmethod
	def add_signal_model(lnpdf, n, prefactors_range, df, inv_snr_pow = 4., snr_min = 3.5, progressbar = None):
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
#                               dt, dphi, deff ratio PDF
#
# =============================================================================
#


def chunker(seq, size, start = None, stop = None):
	"""
	A generator to break up a sequence into chunks of length size, plus the
	remainder, e.g.,

	>>> for x in chunker([1, 2, 3, 4, 5, 6, 7], 3):
	...     print x
	...
	[1, 2, 3]
	[4, 5, 6]
	[7]
	"""
	if start is None:
		start = 0
	if stop is None:
		stop = len(seq)
	return (seq[pos:pos + size] for pos in xrange(start, stop, size))


def normsq_along_one(x):
	"""
	Compute the dot product squared along the first dimension in a fast
	way.  Only works for real floats and numpy arrays.  Not really for
	general use.

	>>> normsq_along_one(numpy.array([[1., 2., 3.], [2., 4., 6.]]))
	array([ 14.,  56.])

	"""
	return numpy.add.reduce(x * x, axis=(1,))


def margprob(Dmat):
	"""
	Compute the marginalized probability along the second dimension of a
	matrix Dmat.  Assumes the probability is the form exp(-x_i^2/2.)

	>>> margprob(numpy.array([[1.,2.,3.,4.], [2.,3.,4.,5.]]))
	[0.41150885406464954, 0.068789418217400547]
	"""

	out = []
	for D in Dmat:
		D = D[numpy.isfinite(D)]
		step = max(int(len(D) / 2048.), 1)
		D = D[::step]
		if len(D) == 2049:
			out.append(step * scipy.integrate.romb(numpy.exp(-D**2/2.)))
		else:
			out.append(step * scipy.integrate.simps(numpy.exp(-D**2/2.)))
	return out


class TimePhaseSNR(object):
	"""
	The goal of this is to compute:

	.. math::

		P(\\vec{\\rho}, \\vec{t}, \\vec{\phi} | \\vec{O}, \\vec{D_H}, s)

	Instead of evaluating the above probability we will parameterize it by
	pulling out an overall term that scales as the network SNR to the negative
	fourth power.

	.. math::

		P(\\vec{\\rho}, \\vec{t}, \\vec{\phi} | \\vec{O}, \\vec{D_H}, s)
		\\approx
		P(\\vec{\\rho_{\mathrm{1\,Mpc}}}, \\vec{t}, \\vec{\phi} | \\vec{O}, \\vec{D_H}, s) \\times |\\vec{\\rho}|^{-4}


	We reduce the dimensionality by computing things relative to the first
	instrument in alphabetical order and use effective distance instead of SNR,
	e.g.,

	.. math::
			P(\\vec{D_{\mathrm{eff}}} / D_{\mathrm{eff}\,0}, \\vec{t} - t_0, \\vec{\phi} - \phi_0 | \\vec{O}, s) 
		\\times
			|\\vec{\\rho}|^{-4} 
		\equiv
			P(\\vec\lambda|\\vec{O}, s)
		\\times
			|\\vec{\\rho}|^{-4}

	where now the absolute geocenter end time or coalescence phase does not
	matter since our new variables are relative to the first instrument in
	alphabetical order (denoted with 0 subscript).  Since the first component of
	each of the new vectors is one by construction we don't track it and have
	reduced the dimensionality by three.

	We won't necessarily evalute exactly this distribution, but something
	that should be proportional to it.  In order to evaluate this distribution. We
	assert that physical signals have a uniform distribution in earth based
	coordinates, :math:`\mathrm{RA}, \cos(\mathrm{DEC}), \cos(\iota), \psi`.  We
	lay down a uniform densly sampled grid in these coordinates and assert that any
	signal should ``exactly" land on one of the grid points. We transform that
	regularly spaced grid into a grid of (irregularly spaced) points in
	:math:`\\vec{\lambda}`, which we denote as :math:`\\vec{\lambda_{\mathrm{m}i}}`
	for the *i*\ th model lambda vector.  We consider that the **only** mechanism
	to push a signal away from one of these exact *i*\ th grid points is Gaussian
	noise.  Furthermore we assume the probability distribution is of the form:

	.. math::

		P(\\vec{\lambda} | \\vec{O}, \\vec{\lambda_{mi}})
		= 
			\\frac{1}{\sqrt{(2\pi)^k |\pmb{\Sigma}|}} 
			\exp{ \left[ -\\frac{1}{2} \\vec{\Delta\lambda}^T \, \pmb{\Sigma}^{-1} \, \\vec{\Delta\lambda} \\right] }

	where :math:`\\vec{\Delta\lambda_i} = \\vec{\lambda} -
	\\vec{\lambda_{\mathrm{m}i}}` and :math:`\pmb{\Sigma}` is diagonal.
	For simplicity here forward we will set :math:`\pmb{\Sigma} = 1` and will drop the normalization, i.e.,

	.. math::

		P(\\vec{\lambda} | \\vec{O}, \\vec{\lambda_{mi}})
		\propto
			\exp{ \left[ -\\frac{1}{2} \\vec{\Delta\lambda_i}^2 \\right] }

	Then we assert that:

	.. math::

		P(\\vec\lambda|\\vec{O}, \\vec{D_H}, s) \propto \sum_i P(\\vec{\lambda} | \\vec{O}, \\vec{\lambda_{mi}}) \, p(\\vec{\lambda_{mi}})

	Where by construction :math:`p(\\vec{\lambda_{mi}})` doesn't depend on
	*i* since they were chosen uniform in prior signal probabality. Computing this
	probability on the fly is tough since the grid might have millions of points.
	Therefore we make another simplifying assumption that the noise only adds a
	contribution which is orthogonal to the hypersurface defined by the signal.  In
	other words, we assume that noise cannot push a signal from one grid point
	towards another along the signal manifold.  In this case we can simplify the
	marginalization step with a precomputation since

	.. math::

		\sum_i P(\\vec{\lambda} |\\vec{O}, \\vec{\lambda_{mi}}) \\approx P(\\vec{\lambda} |\\vec{O}, \\vec{\lambda_{m0}}) \\times \sum_{i > 0} \exp{ \left[ -\\frac{1}{2} \\vec{\Delta x_i}^2 \\right] }


	Where :math:`\\vec{\lambda_{m0}}` is the **nearest neighbor** to the
	measured :math:`\\vec{\lambda}`.  The :math:`\\vec{\Delta x_i}^2` term is the
	distance squared for the *i*\ th grid point and the nearest neighbor point 0.
	This entire sum can be precomputed and stored.

	The geometric interpretation is shown in the following figure:

	.. image:: ../images/TimePhaseSNR01.png
	   :width: 400px

	In order for this endeavor to be successful, we still need a fast way
	to find the nearest neighbor. We use scipy KDTree to accomplish this.


	"""
	# NOTE to save a couple hundred megs of ram we do not
	# include kagra for now...
	responses = {"H1": lal.CachedDetectors[lal.LHO_4K_DETECTOR].response, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].response, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].response}#, "K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].response}
	locations = {"H1":lal.CachedDetectors[lal.LHO_4K_DETECTOR].location, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].location, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].location}#, "K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].location}
	numchunks = 20

	# FIXME compute this more reliably or expose it as a property
	# or something
	sigma = {"time": 0.001, "phase": numpy.pi / 6, "deff": 0.2}

	def __init__(self, tree_data = None, margsky = None, verbose = False, margstart = 0, margstop = None):
		"""
		Initialize a new class from scratch via explicit computation
		of the tree data and marginalized probability distributions or by providing
		these.  **NOTE** generally speaking a user will not initialize
		one of these from scratch, but instead will read the data from disk using the
		from_hdf() method below.
		"""

		self.norm = (4 * numpy.pi**2)**2
		self.tree_data = tree_data
		self.margsky = margsky

		if self.tree_data is None:
			time, phase, deff = TimePhaseSNR.tile(verbose = verbose)
			self.tree_data = self.dtdphideffpoints(time, phase, deff, self.slices)

		# produce KD trees for all the combinations.  NOTE we slice
		# into the same array for memory considerations.  the KDTree
		# does *not* make copies of the data so be careful to not
		# modify it afterward
		self.KDTree = {}
		for combo in self.combos:
			if verbose:
				print >> sys.stderr, "initializing tree for: ", combo
			slcs = sorted(sum(self.instrument_pair_slices(self.instrument_pairs(combo)).values(),[]))
			self.KDTree[combo] = spatial.cKDTree(self.tree_data[:,slcs])

		# NOTE: This is super slow we have a premarginalized h5 file in
		# the tree, see the helper class InspiralExtrinsics
		if self.margsky is None:
			self.margsky = {}
			for combo in self.combos:
				if verbose:
					print >> sys.stderr, "marginalizing tree for: ", combo
				slcs = sorted(sum(self.instrument_pair_slices(self.instrument_pairs(combo)).values(),[]))
				num_points = self.tree_data.shape[0]

				marg = numpy.zeros(num_points)
				for cnt, points in enumerate(chunker(self.tree_data[:,slcs], self.numchunks, margstart, margstop)):
					if verbose:
						print >> sys.stderr, "%d/%d" % (cnt * self.numchunks, num_points)
					Dmat = self.KDTree[combo].query(points, k=num_points, distance_upper_bound = 8.5)[0]
					marg[margstart + self.numchunks * cnt : margstart + self.numchunks * (cnt+1)] = margprob(Dmat)
				self.margsky[combo] = numpy.array(marg, dtype="float32")

	def to_hdf5(self, fname):
		"""
		If you have initialized one of these from scratch and want to
		save it to disk, do so.
		"""
		f = h5py.File(fname, "w")
		dgrp = f.create_group("gstlal_extparams")
		dgrp.create_dataset("treedata", data = self.tree_data, compression="gzip")
		mgrp = dgrp.create_group("marg")
		for combo in self.combos:
			mgrp.create_dataset(",".join(combo), data = self.margsky[combo], compression="gzip")
		f.close()

	@staticmethod
	def from_hdf5(fname, other_fnames = []):
		"""
		Initialize one of these from a file instead of computing it from scratch
		"""
		f = h5py.File(fname, "r")
		dgrp = f["gstlal_extparams"]
		tree_data = numpy.array(dgrp["treedata"])
		margsky = {}
		for combo in dgrp["marg"]:
			key = tuple(combo.split(","))
			margsky[key] = numpy.array(dgrp["marg"][combo])
		f.close()
		for fn in other_fnames:
			f = h5py.File(fn, "r")
			dgrp = f["gstlal_extparams"]
			for combo in dgrp["marg"]:
				key = tuple(combo.split(","))
				margsky[key] += numpy.array(dgrp["marg"][combo])
			f.close()
		return TimePhaseSNR(tree_data = tree_data, margsky = margsky)

	@property
	def combos(self):
		"""
		return instrument combos for all the instruments internally stored in self.responses

		>>> TimePhaseSNR.combos
		(('H1', 'L1'), ('H1', 'L1', 'V1'), ('H1', 'V1'), ('L1', 'V1'))
		"""
		return self.instrument_combos(self.responses)

	@property
	def pairs(self):
		"""
		Return all possible pairs of instruments for the
		instruments internally stored in self.responses
		>>> TimePhaseSNR.pairs
		(('H1', 'L1'), ('H1', 'V1'), ('L1', 'V1'))
		"""

		out = []
		for combo in self.combos:
			out.extend(self.instrument_pairs(combo))
		return tuple(sorted(list(set(out))))

	@property
	def slices(self):
		"""
		This provides a way to index into the internal tree data for
		the delta T, delta phi, and deff ratios for each instrument pair.

		>>> TimePhaseSNR.slices
		{('H1', 'L1'): [0, 1, 2], ('H1', 'V1'): [3, 4, 5], ('L1', 'V1'): [6, 7, 8]}
		"""
		# we will define indexes for slicing into a subset of instrument data
		return dict((pair, [3*n,3*n+1,3*n+2]) for n,pair in enumerate(self.pairs))

	def instrument_pair_slices(self, pairs):
		"""
		define slices into tree data for a given set of instrument
		pairs (which is possibly a subset of the full availalbe pairs)
		"""
		s = self.slices
		return dict((pair, s[pair]) for pair in pairs)

	@classmethod
	def instrument_combos(cls, instruments, min_instruments = 2):
		"""
		Given a list of instrument produce all the possible combinations of min_instruents or greater, e.g.,

		>>> TimePhaseSNR.instrument_combos(("H1","V1","L1"), min_instruments = 3)
		(('H1', 'L1', 'V1'),)
		>>> TimePhaseSNR.instrument_combos(("H1","V1","L1"), min_instruments = 2)
		(('H1', 'L1'), ('H1', 'L1', 'V1'), ('H1', 'V1'), ('L1', 'V1'))
		>>> TimePhaseSNR.instrument_combos(("H1","V1","L1"), min_instruments = 1)
		(('H1',), ('H1', 'L1'), ('H1', 'L1', 'V1'), ('H1', 'V1'), ('L1',), ('L1', 'V1'), ('V1',))

		**NOTE**: these combos are always returned in alphabetical order

		"""

		combos = set()
		# FIXME this probably should be exposed, but 1 doesn't really make sense anyway
		for i in range(min_instruments, len(instruments,) + 1):
			for choice in itertools.combinations(instruments, i):
				# NOTE the reference IFO is always the first in
				# alphabetical order for any given combination,
				# hence the sort here
				combos.add(tuple(sorted(choice)))
		return tuple(sorted(list(combos)))

	def instrument_pairs(self, instruments):
		"""
		Given a list of instruments, construct all possible pairs

		>>> TimePhaseSNR.instrument_pairs(("H1","K1","V1","L1"))
		(('H1', 'K1'), ('H1', 'L1'), ('H1', 'V1'))

		**NOTE**: These are always in alphabetical order
		"""
		out = []
		instruments = tuple(sorted(instruments))
		for i in instruments[1:]:
			out.append((instruments[0], i))
		return tuple(out)

	def dtdphideffpoints(self, time, phase, deff, slices):
		"""
		Given a dictionary of time, phase and deff, which could be
		lists of values, pack the delta t delta phi and eff distance
		ratios divided by the values in self.sigma into an output array according to
		the rules provided by slices.

		>>> TimePhaseSNR.dtdphideffpoints({"H1":0, "L1":-.001, "V1":.001}, {"H1":0, "L1":0, "V1":1}, {"H1":1, "L1":3, "V1":4}, TimePhaseSNR.slices)
		array([[ 1.        ,  0.        , -5.        , -1.        , -2.54647899,
			-5.        , -2.        , -2.54647899, -5.        ]], dtype=float32)

		**NOTE** You must have the same ifos in slices as in the time,
		phase, deff dictionaries.  The result of self.slices gives slices for all the
		instruments stored in self.responses
		"""
		# order is dt, dphi and effective distance ratio for each combination
		# NOTE the instruments argument here really needs to come from calling instrument_pairs()
		if hasattr(time.values()[0], "__iter__"):
			outlen = len(time.values()[0])
		else:
			outlen =1
		out = numpy.zeros((outlen, 1 + max(sum(slices.values(),[]))), dtype="float32")

		for ifos, slc in slices.items():
			ifo1, ifo2 = ifos
			out[:,slc[0]] = (time[ifo1] - time[ifo2]) / self.sigma["time"]
			out[:,slc[1]] = (phase[ifo1] - phase[ifo2]) / self.sigma["phase"]
			# FIXME should this be the ratio - 1 or without the -1 ???
			out[:,slc[2]] = (deff[ifo1] / deff[ifo2] - 1) / self.sigma["deff"]

		return out

	@classmethod
	def tile(cls, NSIDE = 16, NANGLE = 33, verbose = False):
		"""
		Tile the sky with equal area tiles defined by the healpix NSIDE
		and NANGLE parameters.  Also tile polarization uniformly and inclination
		uniform in cosine.  Convert these sky coordinates to time, phase and deff for
		each instrument in self.responses.  Return the sky tiles in the detector
		coordinates as dictionaries.  The default values have millions
		of points in the 4D grid
		"""
		# FIXME should be put at top, but do we require healpy?  It
		# isn't necessary for running at the moment since cached
		# versions of this will be used.
		import healpy
		healpix_idxs = numpy.arange(healpy.nside2npix(NSIDE))
		# We are concerned with a shell on the sky at some fiducial
		# time (which simply fixes Earth as a natural coordinate
		# system)
		T = lal.LIGOTimeGPS(0)
		GMST = lal.GreenwichMeanSiderealTime(T)
		D = 1.
		phase = dict((ifo, numpy.zeros(len(healpix_idxs) * NANGLE**2, dtype="float32")) for ifo in cls.responses)
		deff = dict((ifo, numpy.zeros(len(healpix_idxs) * NANGLE**2, dtype="float32")) for ifo in cls.responses)
		time = dict((ifo, numpy.zeros(len(healpix_idxs) * NANGLE**2, dtype="float32")) for ifo in cls.responses)

		if verbose:
			print >> sys.stderr, "tiling sky: \n"
		cnt = 0
		for i in healpix_idxs:
			if verbose:
				print >> sys.stderr, "sky %04d of %04d\r" % (i, len(healpix_idxs)),
			DEC, RA = healpy.pix2ang(NSIDE, i)
			DEC -= numpy.pi / 2
			for COSIOTA in numpy.linspace(-1, 1, NANGLE):
				hplus = 0.5 * (1.0 + COSIOTA**2)
				hcross = COSIOTA
				for PSI in numpy.linspace(0, numpy.pi * 2, NANGLE):
					for ifo in cls.responses:
						Fplus, Fcross = lal.ComputeDetAMResponse(cls.responses[ifo], RA, DEC, PSI, GMST)
						phase[ifo][cnt] = -numpy.arctan2(Fcross * hcross, Fplus * hplus)
						deff[ifo][cnt] = D / ((Fplus * hplus)**2 + (Fcross * hcross)**2)**0.5
						time[ifo][cnt] = lal.TimeDelayFromEarthCenter(cls.locations[ifo], RA, DEC, T)
					cnt += 1

		if verbose:
			print >> sys.stderr, "\n...done"
		return time, phase, deff

	def __call__(self, time, phase, snr, horizon):
		"""
		Compute the probability of obtaining time, phase and SNR values
		for the instruments specified by the input dictionaries.  We also need the
		horizon distance because we convert to effective distance internally.

		>>> TimePhaseSNR({"H1":0, "L1":0.001, "V1":0.001}, {"H1":0., "L1":1., "V1":1.}, {"H1":5., "L1":7., "V1":7.}, {"H1":1., "L1":1., "V1":1.})
		array([  9.51668418e-14], dtype=float32)

		"""
		deff = dict((k, horizon[k] / snr[k] * 8.0) for k in snr)
		# FIXME can this be a function call??
		slices = dict((pair, [3*n,3*n+1,3*n+2]) for n,pair in enumerate(self.instrument_pairs(time)))
		point = self.dtdphideffpoints(time, phase, deff, slices)
		combo = tuple(sorted(time))
		treedataslices = sorted(sum(self.instrument_pair_slices(self.instrument_pairs(combo)).values(),[]))
		nearestix = self.KDTree[combo].query(point)[1]
		nearestpoint = self.tree_data[nearestix, treedataslices]
		D = (point - nearestpoint)[0]
		D2 = numpy.dot(D,D)
		# FIXME 4. / (sum(s**2 for s in S.values())**.5)**4 is the term
		# that goes like rho^-4 with a somewhat arbitrary normilization
		# which comes form 5.66 ~ (4**2 + 4**2)**.5, so that the factor
		# is 1 for a double right at threshold.
		return numpy.exp(-D2 / 2.) * self.margsky[combo][nearestix] / self.norm * 5.66 / (sum(s**2 for s in snr.values())**.5)**4


#
# =============================================================================
#
#                               P(Ifos | Horizons)
#
# =============================================================================
#


class p_of_instruments_given_horizons(object):
	"""
	The goal of this class is to compute :math:`P(\\vec{O} | \\vec{D_H},
	s)`.  In order to compute it, the SNR is calculated for an ideal signal as a
	function of given sky location and distance and then integrated over the
	extrinsic parameters to figure out the probability that a signal produces and
	above SNR event in each of the :math:`\\vec{O}` detectors.  This probability is
	computed for many possible horizon distance combinations.
	"""
	def __init__(self, instruments = ("H1", "L1", "V1"), snr_thresh = 4., nbins = 41, hmin = 0.05, hmax = 20.0, histograms = None):
		"""
		for each sub-combintation of the "on" instruments, e.g.,
		"H1","L1" out of "H1","L1","V1", the probability of getting a trigger above the
		snr_thresh in each of e.g., "H1", "L1" is computed for different horizon
		distance ratios bracketed from hmin to hmax in nbins per dimension.  Horizon
		distance ratios form an N-1 dimensional function where N is the number of
		instruments in the sub combination.  Thus computing the probability of a triple
		coincidence detection requires a two dimensional histogram with nbins^2 mnumber
		of points.  The probability is interpolated over the bins with linear
		interpolation.

		NOTE! This is a very slow class to initialize from scratch in
		normal circumstances you would use the from_hdf5() method to load precomputed
		values.  NOTE the :py:class`InspiralExtrinsics` provides a helper class to load
		precomputed data.
		"""
		self.instruments = tuple(sorted(list(instruments)))
		self.snr_thresh = snr_thresh
		self.nbins = nbins
		self.hmin = hmin
		self.hmax = hmax
		# NOTE should be sorted

		if histograms is not None:
			self.histograms = histograms
			# NOTE we end up pushing any value outside of our
			# histogram to just be the value in the last(first)
			# bin, so we track those center values here.  
			self.first_center = histograms.values()[0].centres()[0][0]
			self.last_center = histograms.values()[0].centres()[0][-1]
		else:
			combos = TimePhaseSNR.instrument_combos(self.instruments, min_instruments = 1)
			self.histograms = {}
			bins = []
			for i in range(len(self.instruments) - 1):
				# There are N-1 possible horizon distance ratio combinations
				bins.append(rate.LogarithmicBins(self.hmin, self.hmax, self.nbins))
			for combo in combos:
				# Each possible sub combination of ifos gets its own histogram.
				self.histograms[combo] = rate.BinnedArray(rate.NDBins(bins))

			# NOTE we end up pushing any value outside of our
			# histogram to just be the value in the last(first)
			# bin, so we track those center values here.  
			self.first_center = histograms.values()[0].centres()[0][0]
			self.last_center = histograms.values()[0].centres()[0][-1]
			# The sky tile resolution here is lower than the
			# TimePhaseSNR calculation, but it seems good enough.
			_, _, deff = TimePhaseSNR.tile(NSIDE = 8, NANGLE = 17)
			alldeff = []
			for v in deff.values():
				alldeff.extend(v)
			# Figure out the min and max effective distances to help with the integration over physical distance
			mindeff = min(alldeff)
			maxdeff = max(alldeff)

			# Iterate over the N-1 horizon distance ratios for all
			# of the instruments that could have produced coincs
			for horizontuple in itertools.product(*[b.centres() for b in bins]):
				horizondict = {}
				# Calculate horizon distances for each of the
				# instruments based on these ratios NOTE by
				# defn the first instrument in alpha order will
				# always have a horizon of 1
				horizondict[self.instruments[0]] = 1.0
				for i, ifo in enumerate(self.instruments[1:]):
					horizondict[ifo] = horizontuple[i]
				snrs = {}
				snrs_above_thresh = {}
				snrs_below_thresh = {}
				prob = []
				for cnt, ifo in enumerate(horizondict):
					# We want to integrate over physical
					# distance with limits set by the min
					# and max effective distance
					LOW = self.hmin * 8. / self.snr_thresh / maxdeff
					HIGH = max(horizontuple + (1,)) * 8. / self.snr_thresh / mindeff
					for D in numpy.linspace(LOW, HIGH, 200):
						# go from horizon distance to an expected SNR
						snrs.setdefault(ifo,[]).extend(8 * horizondict[ifo] / (D * deff[ifo]))
						# We store the probability
						# associated with this
						# distance, but we only need to
						# do it the first time through
						if cnt == 0:
							prob.extend([D**2] * len(deff[ifo]))
					# Modify the the SNR by a chi
					# distribution with two degrees of
					# freedom.
					snrs[ifo] = stats.ncx2.rvs(2, numpy.array(snrs[ifo])**2)**.5
					snrs_above_thresh[ifo] = snrs[ifo] >= self.snr_thresh
					snrs_below_thresh[ifo] = snrs[ifo] < self.snr_thresh
					prob = numpy.array(prob)
				total = 0.
				# iterate over all subsets
				for combo in combos:
					for cnt, ifo in enumerate(combo):
						if cnt == 0:
							must_be_above = snrs_above_thresh[ifo].copy()
						else:
							must_be_above &= snrs_above_thresh[ifo]
					# the ones above thresh must be accompanied with the compliment to this combo being below thresh
					for ifo in set(self.instruments) - set(combo):
						must_be_above &= snrs_below_thresh[ifo]
					# sum up the prob
					count = sum(prob[must_be_above])
					# record this probability in the histograms
					self.histograms[combo][horizontuple] += count
					total += count
				# normalize the result so that the sum at this horizon ratio is one over all the combinations
				for I in self.histograms:
					self.histograms[I][horizontuple] /= total
		self.mkinterp()

	def mkinterp(self):
		"""
		Create an interpolated represenation over the grid of horizon ratios
		"""
		self.interps = {}
		for I in self.histograms:
			self.interps[I] = rate.InterpBinnedArray(self.histograms[I])

	def __call__(self, instruments, horizon_distances):
		"""
		Calculate the probability of getting a trigger in instruments given the horizon distances.
		"""
		H = [horizon_distances[k] for k in sorted(horizon_distances)]
		return self.interps[tuple(sorted(instruments))](*[min(max(h / H[0], self.first_center), self.last_center) for h in H[1:]])

	def to_hdf5(self, fname):
		"""
		Record the class data to a file so that you don't have to remake it from scratch
		"""
		f = h5py.File(fname, "w")
		grp = f.create_group("gstlal_p_of_instruments")
		grp.attrs["snr_thresh"] = self.snr_thresh
		grp.attrs["hmin"] = self.hmin
		grp.attrs["hmax"] = self.hmax
		grp.attrs["nbins"] = self.nbins
		grp.attrs["instruments"] = ",".join(self.instruments)
		for combo in self.histograms:
			grp.create_dataset(",".join(combo), data = self.histograms[combo].array, compression="gzip")
		f.close()

	@staticmethod
	def from_hdf5(fname):
		"""
		Read data from a file so that you don't have to remake it from scratch
		"""
		f = h5py.File(fname, "r")
		grp = f["gstlal_p_of_instruments"]
		snr_thresh = grp.attrs["snr_thresh"]
		hmin = grp.attrs["hmin"]
		hmax = grp.attrs["hmax"]
		nbins = grp.attrs["nbins"]
		instruments = tuple(sorted(grp.attrs["instruments"].split(",")))
		histograms = {}
		bins = []
		for i in range(len(instruments) - 1):
			bins.append(rate.LogarithmicBins(hmin, hmax, nbins))
		for combo in TimePhaseSNR.instrument_combos(instruments, min_instruments = 1):
			histograms[combo] = rate.BinnedArray(rate.NDBins(bins))
			histograms[combo].array[:] = numpy.array(grp[",".join(combo)])[:]
		f.close()
		return p_of_instruments_given_horizons(instruments = instruments, snr_thresh = snr_thresh, nbins = nbins, hmin = hmin, hmax = hmax, histograms = histograms)


#
# =============================================================================
#
#      Helper class to wrap dt, dphi, deff ratio PDF and P(Ifos | Horizons)
#
# =============================================================================
#


class InspiralExtrinsics(object):
	"""
	Helper class to use preinitialized data for the extrinsic parameter
	calculation. Presently only H,L,V is supported. K could be added by making new
	data files with :py:class:`TimePhaseSNR` and :py:class:`p_of_instruments_given_horizons`.

	This class is used to compute p_of_instruments_given_horizons
	and the probability of getting time phase and snrs from 
	a given instrument combination.  The argument min_instruments will be
	used to normalize the p_of_instruments_given_horizons to set the probability of
	a combination with fewer than min_instruments to be 0.

	>>> IE = InspiralExtrinsics()
	>>> IE.p_of_instruments_given_horizons(("H1","L1"), {"H1":200, "L1":200})
	0.36681567679586446
	>>> IE.p_of_instruments_given_horizons(("H1","L1"), {"H1":20, "L1":200})
	0.0021601523270060085
	>>> IE.p_of_instruments_given_horizons(("H1","L1"), {"H1":200, "L1":200, "V1":200})
	0.14534898937680402

	>>> IE.time_phase_snr({"H1":0.001, "L1":0.0, "V1":0.004}, {"H1":1.3, "L1":4.6, "V1":5.3}, {"H1":20, "L1":20, "V1":4}, {"H1":200, "L1":200, "V1":50})
	array([  1.01240596e-06], dtype=float32)
	>>> IE.time_phase_snr({"H1":0.001, "L1":0.0, "V1":0.004}, {"H1":1.3, "L1":1.6, "V1":5.3}, {"H1":20, "L1":20, "V1":4}, {"H1":200, "L1":200, "V1":50})
	array([  1.47201028e-15], dtype=float32)

	The total probability would be the product, e.g.,

	>>> IE.time_phase_snr({"H1":0.001, "L1":0.0, "V1":0.004}, {"H1":1.3, "L1":4.6, "V1":5.3}, {"H1":20, "L1":20, "V1":8}, {"H1":200, "L1":200, "V1":200}) * IE.p_of_instruments_given_horizons(("H1","L1","V1"), {"H1":200, "L1":200, "V1":200})
	array([  2.00510986e-08], dtype=float32)

	See the following for more details:

	* :py:class:`TimePhaseSNR`
	* :py:class:`p_of_instruments_given_horizons`
	"""
	time_phase_snr = TimePhaseSNR.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "inspiral_dtdphi_pdf.h5"))
	p_of_ifos = {}
	# FIXME add Kagra
	p_of_ifos[("H1", "L1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1L1V1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("H1", "L1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1L1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("H1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1V1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("L1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "L1V1_p_of_instruments_given_H_d.h5"))

	def __init__(self, min_instruments = 1):
		#
		# NOTE every instance will repeat this min_instruments
		# normalization.  Therefore different min_instruments are not
		# supported in the same process. You will get incorrect
		# results.
		#
		self.min_instruments = min_instruments
		# remove combinations less than min instruments and renormalize
		for pofI in self.p_of_ifos.values():
			total = numpy.zeros(pofI.histograms.values()[0].array.shape)
			for combo in list(pofI.histograms.keys()):
				if len(combo) < self.min_instruments:
					del pofI.histograms[combo]
				else:
					total += pofI.histograms[combo].array
			for combo in pofI.histograms:
				pofI.histograms[combo].array /= total
			pofI.mkinterp()

	def p_of_instruments_given_horizons(self, instruments, horizons):
		horizons = dict((k,v) for k,v in horizons.items() if v != 0)
		# if only one instrument is on the probability is 1.0
		if set(horizons) & set(instruments) != set(instruments):
			raise ValueError("A trigger exists for a detector with zero horizon distance")
		if len(horizons) == 1:
			return 1.0
		on_ifos = tuple(sorted(horizons.keys()))
		return self.p_of_ifos[on_ifos](instruments, horizons)
