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

from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
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

.. |O2_O3_LR_ROC| image:: ../images/O2_O3_LR_ROC.png
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

ROC Curves for HL Analysis with the O2 and O3 code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the code here

- https://git.ligo.org/lscsoft/gstlal/raw/master/gstlal-inspiral/tests/lv_stat_test

we generated ROC curves for discriminiting a synthetic signal and noise model
in H and L.  Three traces are shown.  First, the the SNR only terms in the
likelihood ratio, which was the situation in O1 Code (red).  Second, the O2
code with the previous implementation of additional dt and dphi terms (green)
and finally the current implementation (blue).  The improvement of the present
implementation is consistent with the above injection results and this further
demonstrates that the reimplementation has "done no harm" to the O2
configuration.

|O2_O3_LR_ROC|

Review Status
-------------

Do no harm check of O2 results (Complete)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Comparing runs before and after (done)
- Checking the probabilities returned by new code and old code to show consistent results (done)

Check of error assumptions (For O3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Calculate theoretical delta T, delta phi and snr ratio values of an O2 injection set.  Then compute same parameters from injections.  The difference between those (in e.g., a scatter plot) should give a sense of the errors on those parameters caused by noise.  (sarah is working on it)
- Eventually use the fisher matrix for the error estimates (chad will do it, but not for O2)


Inclusion of virgo (Partially complete, rest for O3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Virgo should not make the analysis worse in an average sense. (done, see https://dcc.ligo.org/LIGO-G1801491)
- Understand cases where / if virgo does downrank a trigger (addressed by below)
- Consider having the likelihood calculation maximize over all trigger subsets (Chad and Kipp will do this for O3)


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
		return cls.from_xml(ligolw_utils.load_fileobj(fileobj, gz = True, contenthandler = cls.LIGOLWContentHandler))


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
		ncparam_per_pf = snr2
		# takes into account the mean depending on noncentrality parameter
		snrchi2 = numpy.outer(snr2 * df * (1.0 + max(pfs)), rcoss)

		arr = numpy.zeros_like(lnpdf.array)
		for pf in pfs:
			if progressbar is not None:
				progressbar.increment()
			arr[snrindices, rcossindices] += gstlalstats.ncx2pdf(snrchi2, df, numpy.array([pf * ncparam_per_pf]).T)

		# convert to counts by multiplying by bin volume, and also
		# multiply by an SNR powr law
		arr[snrindices, rcossindices] *= numpy.outer(dsnr / snr**inv_snr_pow, drcoss)

		# normalize to a total count of n
		arr *= n / arr.sum()

		# add to lnpdf
		lnpdf.array += arr

	@staticmethod
	def add_glitch_model(lnpdf, n, prefactors_range, df, inv_snr_pow = 4., snr_min = 3.5, progressbar = None):
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
		step = max(int(len(D) / 65536.), 1)
		D = D[::step]
		if len(D) == 65537:
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
	responses = {"H1": lal.CachedDetectors[lal.LHO_4K_DETECTOR].response, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].response, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].response, "K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].response}
	locations = {"H1":lal.CachedDetectors[lal.LHO_4K_DETECTOR].location, "L1":lal.CachedDetectors[lal.LLO_4K_DETECTOR].location, "V1":lal.CachedDetectors[lal.VIRGO_DETECTOR].location, "K1":lal.CachedDetectors[lal.KAGRA_DETECTOR].location}
	numchunks = 20

	def __init__(self, transtt = None, transtp = None, transpt = None, transpp = None, transdd = None, norm = None, tree_data = None, margsky = None, verbose = False, margstart = 0, margstop = None):
		"""
		Initialize a new class from scratch via explicit computation
		of the tree data and marginalized probability distributions or by providing
		these.  **NOTE** generally speaking a user will not initialize
		one of these from scratch, but instead will read the data from disk using the
		from_hdf() method below.

		transtt, transtp, transpt, transpp, transdd are required.  They
		can be produced by running gstlal_inspiral_compute_dtdphideff_cov_matrix.  An
		example is here

		gstlal_inspiral_compute_dtdphideff_cov_matrix --psd-xml share/O3/2019-05-09-H1L1V1psd_new.xml.gz --H-snr 5 --L-snr 7.0 --V-snr 2.25

		transtt = {frozenset(['V1', 'H1']): 1721.2939671945821, frozenset(['H1', 'L1']): 4264.931381497161, frozenset(['V1', 'L1']): 1768.5971596859304}
		transtp = {frozenset(['V1', 'H1']): -1.620591401760186, frozenset(['H1', 'L1']): -3.07346071253937, frozenset(['V1', 'L1']): -1.7020167980031902}
		transpt = {frozenset(['V1', 'H1']): 0.0, frozenset(['H1', 'L1']): 0.0, frozenset(['V1', 'L1']): 0.0}
		transpp = {frozenset(['V1', 'H1']): 1.2422379329336048, frozenset(['H1', 'L1']): 2.6540061786001834, frozenset(['V1', 'L1']): 1.2984050700516363}
		transdd = {frozenset(['V1', 'H1']): 2.0518233866439894, frozenset(['H1', 'L1']): 4.068667356033675, frozenset(['V1', 'L1']): 2.1420642628918447}

		norm is required. An example is:

		norm = {('H1', 'V1'):332.96168414700816, ('H1', 'L1', 'V1'):7.729877009864116, ('L1', 'V1'):313.34679951193306, ('L1',):0.0358423687136922, ('H1', 'L1'):409.06489455239137, ('H1',):0.0358423687136922, ('V1',):0.0358423687136922}

		typically a user would make a new inspiral_dt_dphi_pdf.h5 file
		which contains all of these by running:

		gstlal_inspiral_create_dt_dphi_snr_ratio_pdfs_dag

		"""

		# This is such that the integral over the sky and over all
		# orientations is 1 for each combo. NOTE!!! the actual
		# probability of getting sources from these combos is not the
		# same. However, that is calculated as part of the
		# p_of_instruments_given_horizon() normalization.
		#
		# If you run this code you will see it is normalized
		#
		# import numpy
		# from gstlal.stats import inspiral_extrinsics
		#
		# TPDPDF = inspiral_extrinsics.InspiralExtrinsics()
		# time, phase, deff = inspiral_extrinsics.TimePhaseSNR.tile(NSIDE = 8, NANGLE = 17)
		# # This actually doesn't matter it is just needed to map from eff distance to
		# # snr, but the TimePhaseSNR code actually undoes this...
		# h = {"H1":1., "L1":1., "V1":1.}
		#
		# combos = TPDPDF.time_phase_snr.combos + (("H1",),("L1",),("V1",))
		#
		# result = dict((k, 0.) for k in combos)
		#
		# def extract_dict(DICT, keys):
		# 	return dict((k,v) for k,v in DICT.items() if k in keys)
		#
		# for i in range(len(time["H1"])):
		# 	t = dict((k, v[i]) for k, v in time.items())
		# 	p = dict((k, v[i]) for k, v in phase.items())
		# 	s = dict((k, h[k] / v[i]) for k, v in deff.items())
		#
		# 	for ifos in combos:
		# 		t2 = extract_dict(t, ifos)
		# 		p2 = extract_dict(p, ifos)
		# 		s2 = extract_dict(s, ifos)
		# 		h2 = extract_dict(h, ifos)
		# 		result[ifos] += TPDPDF.time_phase_snr(t2,p2,s2,h2) * (sum(x**2 for x in s2.values())**.5)**4 / len(time["H1"]) * 1. / (16. * numpy.pi**2)
		#
		# print result
		# >>> {('H1', 'V1'): array([ 1.00000003]), ('H1', 'L1', 'V1'): array([ 1.00000004]), ('L1', 'V1'): array([ 0.99999999]), ('L1',): 0.99999999999646638, ('H1', 'L1'): array([ 0.99999997]), ('H1',): 0.99999999999646638, ('V1',): 0.99999999999646638}

		if any([x is None for x in (norm, transtt, transtp, transpt, transpp, transdd)]):
			raise ValueError("transtt, transtp, transpt, transpp, transdd and norm are required and cannot be None")

		self.norm = norm
		self.transtt = transtt
		self.transtp = transtp
		self.transpt = transpt
		self.transpp = transpp
		self.transdd = transdd

		self.tree_data = tree_data
		self.margsky = margsky

		if self.tree_data is None or numpy.shape(self.tree_data)[1] != 1 + max(sum(self.slices.values(),[])):
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
					Dmat = self.KDTree[combo].query(points, k=num_points, distance_upper_bound = 15.0)[0]
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

		h5_transtt = f.create_group("transtt")
		h5_transtp = f.create_group("transtp")
		h5_transpt = f.create_group("transpt")
		h5_transpp = f.create_group("transpp")
		h5_transdd = f.create_group("transdd")
		h5_norm = f.create_group("norm")
		for group, mat in zip((h5_transtt, h5_transtp, h5_transpt, h5_transpp, h5_transdd, h5_norm), (self.transtt, self.transtp, self.transpt, self.transpp, self.transdd, self.norm)):
			for k,v in mat.items():
				group.create_dataset(",".join(sorted(k)), data = float(v))

		f.close()

	@staticmethod
	def from_hdf5(fname, other_fnames = [], **kwargs):
		"""
		Initialize one of these from a file instead of computing it from scratch
		"""
		f = h5py.File(fname, "r")
		if os.path.join(gstlal_config_paths["pkgdatadir"], "covmat.h5") in other_fnames:
			# These *have* to be here
			f_covmat = h5py.File(os.path.join(gstlal_config_paths["pkgdatadir"], "covmat.h5"))
			other_fnames.remove(os.path.join(gstlal_config_paths["pkgdatadir"], "covmat.h5"))
			transtt = dict((frozenset(k.split(",")), numpy.array(f_covmat["transtt"][k])) for k in f_covmat["transtt"])
			transtp = dict((frozenset(k.split(",")), numpy.array(f_covmat["transtp"][k])) for k in f_covmat["transtp"])
			transpt = dict((frozenset(k.split(",")), numpy.array(f_covmat["transpt"][k])) for k in f_covmat["transpt"])
			transpp = dict((frozenset(k.split(",")), numpy.array(f_covmat["transpp"][k])) for k in f_covmat["transpp"])
			transdd = dict((frozenset(k.split(",")), numpy.array(f_covmat["transdd"][k])) for k in f_covmat["transdd"])
			norm = dict((frozenset(k.split(",")), numpy.array(f_covmat["norm"][k])) for k in f_covmat["norm"])
		else:
			# These *have* to be here
			transtt = dict((frozenset(k.split(",")), numpy.array(f["transtt"][k])) for k in f["transtt"])
			transtp = dict((frozenset(k.split(",")), numpy.array(f["transtp"][k])) for k in f["transtp"])
			transpt = dict((frozenset(k.split(",")), numpy.array(f["transpt"][k])) for k in f["transpt"])
			transpp = dict((frozenset(k.split(",")), numpy.array(f["transpp"][k])) for k in f["transpp"])
			transdd = dict((frozenset(k.split(",")), numpy.array(f["transdd"][k])) for k in f["transdd"])
			norm = dict((frozenset(k.split(",")), numpy.array(f["norm"][k])) for k in f["norm"])

		try:
			dgrp = f["gstlal_extparams"]
			tree_data = numpy.array(dgrp["treedata"])
			margsky = {}
			for combo in dgrp["marg"]:
				key = tuple(combo.split(","))
				margsky[key] = numpy.array(dgrp["marg"][combo])
		except:
			tree_data = None
			margsky = None

		f.close()

		# FIXME add sanity checking on this - also make it a += ? In
		# general this is kinda crappy.
		for fn in other_fnames:
			f = h5py.File(fn, "r")
			dgrp = f["gstlal_extparams"]
			for combo in dgrp["marg"]:
				key = tuple(combo.split(","))
				margsky[key] += numpy.array(dgrp["marg"][combo])
			f.close()

		return TimePhaseSNR(transtt = transtt, transtp = transtp, transpt = transpt, transpp = transpp, transdd = transdd, norm = norm, tree_data = tree_data, margsky = margsky, **kwargs)

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
		ratios transformed by the covariance matrix according to
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
			dt = (time[ifo1] - time[ifo2])
			dphi = (phase[ifo1] - phase[ifo2])
			# FIXME precompute?
			dtdphivec = numpy.array([dt, dphi])
			coordtransmat = numpy.array([[self.transtt[frozenset((ifo1, ifo2))], self.transtp[frozenset((ifo1, ifo2))]],[self.transpt[frozenset((ifo1, ifo2))], self.transpp[frozenset((ifo1, ifo2))]]])
			out[:,slc[0]], out[:,slc[1]] = numpy.dot(coordtransmat, dtdphivec)
			# FIXME should this be the ratio - 1 or without the -1 ???
			out[:,slc[2]] = numpy.log(deff[ifo1] / deff[ifo2]) * self.transdd[frozenset((ifo1, ifo2))]

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
		combo = tuple(sorted(time))
		#
		# NOTE shortcut for single IFO
		#
		if len(snr) == 1:
			return 1. /  self.norm[frozenset(combo)] * 5.66 / (sum(s**2 for s in snr.values())**.5)**4

		deff = dict((k, horizon[k] / snr[k] * 8.0) for k in snr)
		# FIXME can this be a function call??
		slices = dict((pair, [3*n,3*n+1,3*n+2]) for n,pair in enumerate(self.instrument_pairs(time)))
		point = self.dtdphideffpoints(time, phase, deff, slices)
		treedataslices = sorted(sum(self.instrument_pair_slices(self.instrument_pairs(combo)).values(),[]))
		nearestix = self.KDTree[combo].query(point)[1]
		nearestpoint = self.tree_data[nearestix, treedataslices]
		D = (point - nearestpoint)[0]
		D2 = numpy.dot(D,D)
		# FIXME 4. / (sum(s**2 for s in S.values())**.5)**4 is the term
		# that goes like rho^-4 with a somewhat arbitrary normilization
		# which comes form 5.66 ~ (4**2 + 4**2)**.5, so that the factor
		# is 1 for a double right at threshold.
		return numpy.exp(-D2 / 2.) * self.margsky[combo][nearestix] / self.norm[frozenset(combo)] * 5.66 / (sum(s**2 for s in snr.values())**.5)**4


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
	extrinsic parameters to figure out the probability that a signal produces an
	above SNR event in each of the :math:`\\vec{O}` detectors.  This
	probability is computed for many possible horizon distance combinations. In
	other words the probability of having H and L participate in a coinc when H, L,
	and V are operating is,

	.. math::

		P(H, L | D_H, D_L, D_V, s) = \int_\Sigma P(\\rho_{H}, \\rho_{L}, \\rho_{V} | D_H, D_L, D_V, s)

	where

	.. math::

		\Sigma \equiv
		\\begin{cases}
			\\rho_H \geq \\rho_m \\\\
			\\rho_H \geq \\rho_m \\\\
			\\rho_V \leq \\rho_m
		\end{cases}

	with :math:`\\rho_m` as the SNR threshold of the pipeline.  We
	construct :math:`P(\\rho_{H},\ldots | \dots)` from random sampling of uniform
	location and orientation sources and according to distance squared.  The
	location / orientation sampling is identical to the one used in
	:py:class:`TimePhaseSNR`.  We add a random jitter to each SNR according to a
	chi-squared distribution with two degrees of freedom.

	The result of this is stored in a histogram, which means we choose quanta of horizon distances to do the calculation. Since we only care about
	the ratios of horizon distances in this calculation, the horizon distance for
	the first detector in alphabetical order are by convention 1.  The ratios of horizon distances for the other detectors are logarithmically spaced between 0.05 and 20.  Below are a couple of example histograms.

	**Example: 1D histogram for** :math:`p(H1,L1 | D_H, D_L)`:

	This is a 1x41 array representing the following probabilities:

	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+
	| :math:`p(H1,L1| D_H = 1, D_L = 0.05)`               | ...        | ...      | :math:`p(H1,L1| D_H = 1, D_L = 19)`                 |
	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+

	Note, linear interpolation is used over the bins

	**Example 2D histogram for** :math:`p(H1,L1,V1 | D_H, D_L, D_V)`:

	This is a 41x41 array representing the following probabilities:

	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+
	| :math:`p(H1,L1,V1| D_H = 1, D_L = 0.05, D_V= 0.05)` | ...        | ...      | :math:`p(H1,L1,V1| D_H = 1, D_L = 19, D_V= 0.05)`   |
	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+
	| :math:`p(H1,L1,V1| D_H = 1, D_L = 0.05, D_V= 0.06)` | ...        | ...      | :math:`p(H1,L1,V1| D_H = 1, D_L = 19, D_V= 0.06)`   |
	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+
	| ...                                                 | ...        | ...      | ...                                                 |
	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+
	| :math:`p(H1,L1,V1| D_H = 1, D_L = 0.05, D_V= 19)`   | ...        | ...      | :math:`p(H1,L1,V1| D_H = 1, D_L = 19, D_V= 19)`     |
	+-----------------------------------------------------+------------+----------+-----------------------------------------------------+

	Note, linear interpolation is used over the bins
	"""
	def __init__(self, instruments = ("H1", "L1", "V1"), snr_thresh = 4., nbins = 41, hmin = 0.05, hmax = 20.0, histograms = None):
		"""
		for each sub-combintation of the "on" instruments, e.g.,
		"H1","L1" out of "H1","L1","V1", the probability of getting a trigger above the
		snr_thresh in each of e.g., "H1", "L1" is computed for different horizon
		distance ratios bracketed from hmin to hmax in nbins per dimension.  Horizon
		distance ratios form an N-1 dimensional function where N is the number of
		instruments in the sub combination.  Thus computing the probability of a triple
		coincidence detection requires a two dimensional histogram with nbins^2 number
		of points.  The probability is interpolated over the bins with linear
		interpolation.

		NOTE! This is a very slow class to initialize from scratch in
		normal circumstances you would use the from_hdf5() method to load precomputed
		values.  NOTE the :py:class`InspiralExtrinsics` provides a helper class to load
		precomputed data.  See its documentation.  In general *always
		use* :py:class`InspiralExtrinsics`
		"""
		# The instruments that are being considered for this calculation
		self.instruments = tuple(sorted(list(instruments)))

		# The SNR threshold above which one declares a *found* trigger.
		self.snr_thresh = snr_thresh

		# The number of bins in the histogram of horizond distance ratios.
		self.nbins = nbins

		# The minimum and maximum horizon distance ratio to consider
		# for the binning.  Anything outside this range will be
		# clipped.
		self.hmin = hmin
		self.hmax = hmax

		# If the user has provided the histograms already, we just use
		# them.  This would be the case if the from_hdf5() method is
		# called
		if histograms is not None:
			self.histograms = histograms
			# NOTE we end up pushing any value outside of our
			# histogram to just be the value in the last(first)
			# bin, so we track those center values here in order to
			# decide if something should be clipped.
			self.first_center = self.histograms.values()[0].centres()[0][0]
			self.last_center = self.histograms.values()[0].centres()[0][-1]
		# Otherwise we need to initialize these ourselves, which can be pretty slow.
		else:
			# We reuse the function in TimePhaseSNR to get
			# combinations of instruments.  We compute
			# probabilities for all instruments including singles.
			# Even if you are running a min_instruments = 2
			# analysis, the helper class  InspiralExtrinsics will
			# renormalize this pdf for you.  So you should never
			# call this directly
			combos = TimePhaseSNR.instrument_combos(self.instruments, min_instruments = 1)

			# Setup the bins needed to store the probabilities of
			# getting a combination of instruments: Since we only
			# care about ratios, this will always be a
			# len(instruments) - 1 dimensional histogram, e.g., 2D
			# for H,L,V.  Obviously this approach won't scale to 5
			# detectors very well and we will need something new,
			# but it should hopefully be okay for 4 detectors to
			# get us through O3,O4,...  We use only 41 bins per
			# dimension, so a 3D histogram still has fewer than
			# 100,000 points.  While a pain to precompute, it is
			# not difficult to store or use.
			self.histograms = {}
			bins = []
			for i in range(len(self.instruments) - 1):
				# There are N-1 possible horizon distance ratio combinations
				bins.append(rate.LogarithmicBins(self.hmin, self.hmax, self.nbins))
			for combo in combos:
				# Each possible sub combination of ifos gets its own histogram.
				self.histograms[combo] = rate.BinnedArray(rate.NDBins(bins))

			# NOTE we end up clipping any value outside of our
			# histogram to just be the value in the last(first)
			# bin, so we track those center values here.
			self.first_center = histograms.values()[0].centres()[0][0]
			self.last_center = histograms.values()[0].centres()[0][-1]

			# Now we start the monte carlo simulation of a bunch of
			# signals distributed uniformly in the volume of space

			# The sky tile resolution here is lower than the
			# TimePhaseSNR calculation, but it seems good enough.
			# The only variable we care about here is deff, which a
			# dictionary of effective distances keyed by IFO for
			# the sky grid produced here assuming a physical
			# distance of 1 (thus deff is strictly >= 1).
			_, _, deff = TimePhaseSNR.tile(NSIDE = 8, NANGLE = 17)

			# Figure out the min and max effective distances to help with the integration over physical distance
			alldeff = []
			for v in deff.values():
				alldeff.extend(v)
			mindeff = min(alldeff)
			maxdeff = max(alldeff)

			# Iterate over the (# of instruments - 1) horizon
			# distance ratios for all of the instruments that could
			# have produced coincs
			for horizontuple in itertools.product(*[b.centres() for b in bins]):
				horizondict = {}
				# Calculate horizon distances for each of the
				# instruments based on these ratios NOTE by
				# convention the first instrument in
				# alphabetical order will always have a horizon
				# of 1 since we are only concerned about the ratio!
				horizondict[self.instruments[0]] = 1.0
				for i, ifo in enumerate(self.instruments[1:]):
					horizondict[ifo] = horizontuple[i]

				#
				# Begin the count of sources, intended to
				# simulate real GWS, which pass a SNR threshold
				# to be found in a given set of detectors for a
				# given set of horizon distance ratios provided
				# by horizondict keyed by ifo
				#
				# 1) We generate a source at each point of a
				# coarse grid on the sky (healpix NSIDE = 8) as
				# well as a coarse gridding in psi and iota (17
				# points each)) for a total of 221952 source
				# locations and orientations
				#
				# 2) We interate over 200 points in distance
				# between a LOW value and a HIGH value taken
				# from the range of effective distances between
				# detectors in the sky grid that should ensure
				# that we properly sample the probability space
				#
				# 3) We calculate the SNR for each distance for
				# each ifo for each sky point by choosing a
				# random number with the calculated SNR as the
				# mean but drawn from a chi2 distribution, in
				# order to model noise.  We track the above and
				# below threshold counts
				#
				# Note: The above samples should all be uniform
				# in the volume of space (and thereby a good
				# prior for GWs).  We actually do a linear in
				# distance sampling but weight it by D^2 to
				# ensure this.
				snrs = {}
				snrs_above_thresh = {}
				snrs_below_thresh = {}
				prob = []
				for cnt, ifo in enumerate(horizondict):
					# We want to integrate over physical
					# distance with limits set by the min
					# and max effective distance
					# FIXME these variables, LOW and HIGH
					# should be moved out of the loop for
					# clarity.  The point is that they are
					# the same for every instrument
					LOW = self.hmin * 8. / self.snr_thresh / maxdeff
					HIGH = max(horizontuple + (1,)) * 8. / self.snr_thresh / mindeff
					for D in numpy.linspace(LOW, HIGH, 200):
						# go from horizon distance to an expected SNR
						# NOTE deff[ifo] is an array of effective distances
						snrs.setdefault(ifo,[]).extend(8 * horizondict[ifo] / (D * deff[ifo]))
						# We store the probability
						# associated with this
						# distance, but we only need to
						# do it the first time through
						if cnt == 0:
							prob.extend([D**2] * len(deff[ifo]))
					# Actually choose the SNR from a chi
					# distribution with two degrees of
					# freedom.  Only use the calculated SNR
					# as the mean.
					snrs[ifo] = stats.ncx2.rvs(2, numpy.array(snrs[ifo])**2)**.5
					snrs_above_thresh[ifo] = snrs[ifo] >= self.snr_thresh
					snrs_below_thresh[ifo] = snrs[ifo] < self.snr_thresh
					prob = numpy.array(prob)

				#
				# Now that we have figured out how many
				# triggers survive above the threshold and how
				# many are below threshold, we can go through
				# and work out the prbabilities that we are
				# after for each combination of IFOs.
				# Normalizing the fractional counts gives us
				# the P(ifos | horizons).
				#
				total = 0.
				# iterate over all subsets
				for combo in combos:
					# All the ifos in this combo must be above threshold.
					for cnt, ifo in enumerate(combo):
						if cnt == 0:
							must_be_above = snrs_above_thresh[ifo].copy()
						else:
							must_be_above &= snrs_above_thresh[ifo]
					# The ones above thresh must be accompanied with the compliment to this combo being below thresh
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
		Calculate the probability of getting a trigger in instruments
		given the horizon distances.  NOTE! this should never be called directly.
		Always use the helper class InspiralExtrinsics in order to deal with different
		possibilities for min_instruments (i.e., running an analysis with singles
		enabled or not)
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

	>>> IE.p_of_instruments_given_horizons(("H1","L1"), {"H1":200, "L1":200, "V1":200}) + IE.p_of_instruments_given_horizons(("H1","V1"), {"H1":200, "L1":200, "V1":200}) + IE.p_of_instruments_given_horizons(("L1","V1"), {"H1":200, "L1":200, "V1":200}) + IE.p_of_instruments_given_horizons(("H1","L1","V1"), {"H1":200, "L1":200, "V1":200})
	1.0

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
	p_of_ifos = {}
	# FIXME add Kagra
	p_of_ifos[("H1", "L1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1L1V1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("H1", "L1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1L1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("H1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "H1V1_p_of_instruments_given_H_d.h5"))
	p_of_ifos[("L1", "V1",)] = p_of_instruments_given_horizons.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "L1V1_p_of_instruments_given_H_d.h5"))

	def __init__(self, min_instruments = 1, filename = None):
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
		if filename is not None:
			self.time_phase_snr = TimePhaseSNR.from_hdf5(filename)
		else:
			self.time_phase_snr = TimePhaseSNR.from_hdf5(os.path.join(gstlal_config_paths["pkgdatadir"], "inspiral_dtdphi_pdf.h5"))


	def p_of_instruments_given_horizons(self, instruments, horizons):
		horizons = dict((k,v) for k,v in horizons.items() if v != 0)
		# if only one instrument is on the probability is 1.0
		if set(horizons) & set(instruments) != set(instruments):
			raise ValueError("A trigger exists for a detector with zero horizon distance")
		if len(horizons) == 1:
			return 1.0
		on_ifos = tuple(sorted(horizons.keys()))
		return self.p_of_ifos[on_ifos](instruments, horizons)

#
# =============================================================================
#
#          NOTE OLD CODE!                       dt dphi PDF
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


