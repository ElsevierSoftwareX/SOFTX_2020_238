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
import multiprocessing
import multiprocessing.queues
import numpy
import random
from scipy import interpolate
from scipy import optimize
import os
import sys
import time


from ligo.lw import ligolw
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import lsctables
from ligo.lw import utils as ligolw_utils
from glue.text_progress_bar import ProgressBar
from lal import rate
from lalburst import snglcoinc
from ligo import segments
from ligo.segments import utils as segmentsUtils


from gstlal import stats as gstlalstats
from gstlal.stats import inspiral_lr


#
# =============================================================================
#
#                              Ranking Statistic
#
# =============================================================================
#


def kwarggeniter(d, min_instruments):
	d = tuple(sorted(d.items()))
	return map(dict, itertools.chain(*(itertools.combinations(d, i) for i in range(min_instruments, len(d) + 1))))


def kwarggen(segments, snrs, chi2s_over_snr2s, phase, dt, template_id, min_instruments):
	# segments and template_id held fixed
	for snrs, chi2s_over_snr2s, phase, dt in zip(
		kwarggeniter(snrs, min_instruments),
		kwarggeniter(chi2s_over_snr2s, min_instruments),
		kwarggeniter(phase, min_instruments),
		kwarggeniter(dt, min_instruments)
	):
		yield {
			"segments": segments,
			"snrs": snrs,
			"chi2s_over_snr2s": chi2s_over_snr2s,
			"phase": phase,
			"dt": dt,
			"template_id": template_id
		}


class RankingStat(snglcoinc.LnLikelihoodRatioMixin):
	ligo_lw_name_suffix = u"gstlal_inspiral_rankingstat"

	#
	# Default content handler for loading RankingStat objects from XML
	# documents
	#

	@ligolw_array.use_in
	@ligolw_param.use_in
	@lsctables.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass

	# network SNR threshold
	# now clustering is enabled
	network_snrsq_threshold = 0

	def __init__(self, template_ids = None, instruments = frozenset(("H1", "L1", "V1")), population_model_file = None, dtdphi_file = None, min_instruments = 1, delta_t = 0.005, horizon_factors = None, idq_file = None):
		self.numerator = inspiral_lr.LnSignalDensity(template_ids = template_ids, instruments = instruments, delta_t = delta_t, population_model_file = population_model_file, dtdphi_file = dtdphi_file, min_instruments = min_instruments, horizon_factors = horizon_factors, idq_file = idq_file)
		self.denominator = inspiral_lr.LnNoiseDensity(template_ids = template_ids, instruments = instruments, delta_t = delta_t, min_instruments = min_instruments)
		self.zerolag = inspiral_lr.LnLRDensity(template_ids = template_ids, instruments = instruments, delta_t = delta_t, min_instruments = min_instruments)

	def fast_path_cut(self, snrs, **kwargs):
		"""
		Return True if the candidate described by kwargs should be
		cut, False otherwise.  Used to fast-path out of the full
		likelihood evaluation, and to drop coincs from the
		coincidence engine to reduce data rate.

		NOTE:  surviving this cut is not an endorsement of the
		candidate, many candidates that survive this cut will
		subsequently be discarded for other reasons.  This code is
		only intended to achieve a computationally efficient data
		rate reduction that does not negatively impact the search
		sensitivity.
		"""
		# network SNR cut
		if sum(snr**2. for snr in snrs.values()) < self.network_snrsq_threshold:
			return True
		return False

	def __call__(self, **kwargs):
		"""
		Evaluate the ranking statistic.
		"""
		# ranking statistic is only defined for SNRs at or above
		# the threshold.  modern gstlal_inspiral generates
		# sub-threshold triggers for Bayestar and we need to be
		# ceratin they don't leak into here.
		assert all(snr >= self.snr_min for snr in kwargs["snrs"].values())

		# fast-path cut
		if self.fast_path_cut(**kwargs):
			return NegInf

		# FIXME NOTE
		# Here we put in a penalty for single detector triggers.
		# This is a tuned parameter.
		lnP = 0. if len(kwargs["snrs"]) > 1 else -8.

		# full ln L ranking stat.  we define the ranking statistic
		# to be the largest ln L from all allowed subsets of
		# triggers. Maximizes over higher than double IFO combos.
		return lnP + super(RankingStat, self).__call__(**kwargs) if len(kwargs["snrs"])==1 else max(super(RankingStat, self).__call__(**kwargs) for kwargs in kwarggen(min_instruments = max(2, self.min_instruments), **kwargs))

	@property
	def template_ids(self):
		return self.denominator.template_ids

	@template_ids.setter
	def template_ids(self, value):
		self.numerator.template_ids = value
		self.denominator.template_ids = value
		self.zerolag.template_ids = value

	@property
	def snr_min(self):
		return self.numerator.snr_min

	@property
	def instruments(self):
		return self.denominator.instruments

	@property
	def min_instruments(self):
		return self.denominator.min_instruments

	@property
	def delta_t(self):
		return self.denominator.delta_t

	@property
	def population_model_file(self):
		return self.numerator.population_model_file

	@property
	def dtdphi_file(self):
		return self.numerator.dtdphi_file

	@property
	def idq_file(self):
		return self.numerator.idq_file

	@property
	def horizon_factors(self):
		return self.numerator.horizon_factors

	@property
	def segmentlists(self):
		return self.denominator.segmentlists

	def __iadd__(self, other):
		if type(other) != type(self):
			raise TypeError(other)
		self.numerator += other.numerator
		self.denominator += other.denominator
		self.zerolag += other.zerolag
		return self

	def copy(self):
		new = type(self)(template_ids = self.template_ids, instruments = self.instruments, population_model_file = self.population_model_file, dtdphi_file = self.dtdphi_file, min_instruments = self.min_instruments, delta_t = self.delta_t)
		new.numerator = self.numerator.copy()
		new.denominator = self.denominator.copy()
		new.zerolag = self.zerolag.copy()
		# NOTE:  only if denominator.lnzerolagdensity is pointing
		# to *our* zero-lag density will the copy's be set,
		# otherwise the copy's will be reset to None
		if self.denominator.lnzerolagdensity is self.zerolag:
			new.denominator.lnzerolagdensity = new.zerolag
		return new

	def kwargs_from_triggers(self, events, offsetvector):
		"""
		Constructs the key-word arguments to be passed to
		.__call__() from a sequence of single-detector triggers
		constituting a coincident candidate collected with the
		given offset vector.  For internal use by the
		*_from_triggers() methods.
		"""
		#
		# exclude triggers that are below the SNR threshold.  this
		# is easier to do here, when what we have is triggers, than
		# in the .__call__() method where their parameters have
		# already been mixed into the kwargs.
		#

		events = tuple(event for event in events if event.snr >= self.snr_min)
		assert len(events) >= self.min_instruments, "coincidence engine failed to respect minimum instrument count requirement for candidates:  found candidate with %d < %d instruments" % (len(events), self.min_instruments)

		#
		# pick a random, but reproducible, trigger to provide a
		# reference timestamp for, e.g, the \Delta t's between
		# instruments and the time spanned by the candidate.
		#
		# the trigger times are conveyed as offsets-from-epoch.
		# the trigger times are taken to be their time-shifted
		# values, the time-shifted reference trigger is used to
		# define the epoch.  the objective here is to allow the
		# trigger times to be converted to floats without loss of
		# precision, without loosing knowledge of the \Delta t's
		# between triggers, and in such a way that singles always
		# have a time-shifted offset-from-epoch of 0.
		#
		# for the time spanned by the event, we need a segment for
		# every instrument whether or not it provided a trigger,
		# and reflecting the offset vector that was considered when
		# this candidate was formed (the ranking statistic needs to
		# know when it was we were looking for triggers in the
		# instruments that failed to provide them).  for
		# instruments that do not provide a trigger, we time-shift
		# the reference trigger's interval under the assumption
		# that because we use exact-match coincidence the interval
		# is the same for all instruments.
		#

		reference = min(events, key = lambda event: event.ifo)
		ref_end, ref_offset = reference.end, offsetvector[reference.ifo]
		template_id = reference.template_id
		if template_id not in self.template_ids:
			raise ValueError("event IDs %s are from the wrong template" % ", ".join(sorted(str(event.event_id) for event in events)))
		# segment spanned by reference event
		seg = segments.segment(ref_end - reference.template_duration, ref_end)
		# initially populate segs dictionary shifting reference
		# instrument's segment according to offset vectors
		segs = dict((instrument, seg.shift(ref_offset - offsetvector[instrument])) for instrument in self.instruments)
		# for any any real triggers we have, use their true
		# intervals
		segs.update((event.ifo, segments.segment(event.end - event.template_duration, event.end)) for event in events)

		# done
		return dict(
			segments = segs,
			snrs = dict((event.ifo, event.snr) for event in events),
			chi2s_over_snr2s = dict((event.ifo, event.chisq / event.snr**2.) for event in events),
			phase = dict((event.ifo, event.coa_phase) for event in events),
			dt = dict((event.ifo, float(event.end - ref_end) + offsetvector[event.ifo] - ref_offset) for event in events),
			template_id = template_id
		)

	def fast_path_cut_from_triggers(self, events, offsetvector):
		"""
		Evaluate the ranking statistic's fast-path cut for a
		sequence of single-detector triggers constituting a
		coincident candidate collected with the given offset
		vector.
		"""
		return self.fast_path_cut(**self.kwargs_from_triggers(events, offsetvector))

	def ln_lr_from_triggers(self, events, offsetvector):
		"""
		Evaluate the ranking statistic for a sequence of
		single-detector triggers constituting a coincident
		candidate collected with the given offset vector.
		"""
		try:
			return self(**self.kwargs_from_triggers(events, offsetvector))
		except (ValueError, AssertionError) as e:
			raise type(e)("%s: event IDs %s, offsets %s" % (str(e), ", ".join(sorted(str(event.event_id) for event in events)), str(offsetvector)))

	def finish(self):
		self.numerator.finish()
		self.denominator.finish()
		self.zerolag.finish()
		return self

	def is_healthy(self, verbose = False):
		# do we believe the PDFs are sufficiently well-defined to
		# compute ln L?  not healthy until at least one instrument
		# in the analysis has produced triggers, and until all that
		# have produced triggers have each produced at least 10
		# million.
		# NOTE:  this will go badly if a detector that has never
		# produced triggers, say because it joins an observing run
		# late, suddenly starts producing triggers between snapshot
		# cycles of an online analysis.  we're assuming, here, that
		# detectors join science runs not at random times, but at
		# scheduled times, say, during maintenance breaks, and that
		# the analysis will not be collecting any candidates for
		# approximately one snapshot interval around the addition
		# of the new detector.
		nonzero_counts = [count for count in self.denominator.triggerrates.counts.values() if count]
		health = 0. if not nonzero_counts else min(nonzero_counts) / 10000000.
		if verbose:
			ProgressBar(text = "ranking stat. health", value = health).show()
		return health >= 1.

	@classmethod
	def get_xml_root(cls, xml, name):
		"""
		Sub-classes can use this in their overrides of the
		.from_xml() method to find the root element of the XML
		serialization.
		"""
		name = u"%s:%s" % (name, cls.ligo_lw_name_suffix)
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		return xml[0]

	@classmethod
	def from_xml(cls, xml, name):
		"""
		In the XML document tree rooted at xml, search for the
		serialized RankingStat object named name, and deserialize
		it.  The return value is a two-element tuple.  The first
		element is the deserialized RankingStat object, the second
		is the process ID recorded when it was written to XML.
		"""
		xml = cls.get_xml_root(xml, name)
		self = cls.__new__(cls)
		self.numerator = inspiral_lr.LnSignalDensity.from_xml(xml, "numerator")
		self.denominator = inspiral_lr.LnNoiseDensity.from_xml(xml, "denominator")
		self.zerolag = inspiral_lr.LnLRDensity.from_xml(xml, "zerolag")
		return self

	def to_xml(self, name):
		"""
		Serialize this RankingStat object to an XML fragment and
		return the root element of the resulting XML tree.
		"""
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.numerator.to_xml("numerator"))
		xml.appendChild(self.denominator.to_xml("denominator"))
		xml.appendChild(self.zerolag.to_xml("zerolag"))
		return xml


class DatalessRankingStat(RankingStat):
	# NOTE:  .__iadd__(), .copy() and I/O are forbidden, but these
	# operations will be blocked by the .numerator and .denominator
	# instances, no need to add extra code here to prevent these
	# operations
	def __init__(self, *args, **kwargs):
		self.numerator = inspiral_lr.DatalessLnSignalDensity(*args, **kwargs)
		kwargs.pop("population_model_file", None)
		kwargs.pop("dtdphi_file", None)
		kwargs.pop("idq_file", None)
		self.denominator = inspiral_lr.DatalessLnNoiseDensity(*args, **kwargs)

	def finish(self):
		# no zero-lag
		self.numerator.finish()
		self.denominator.finish()
		return self

	def is_healthy(self, verbose = False):
		if verbose:
			ProgressBar(text = "ranking stat. health", value = 1.).show()
		return True


class OnlineFrankensteinRankingStat(RankingStat):
	"""
	Version of RankingStat with horizon distance history and trigger
	rate history spliced in from another instance.  Used to solve a
	chicken-or-egg problem and assign ranking statistic values in an
	aonline anlysis.  NOTE:  the donor data is not copied, instances of
	this class hold references to the donor's data, so as it is
	modified those modifications are immediately reflected here.

	For safety's sake, instances cannot be written to or read from
	files, cannot be marginalized together with other instances, nor
	accept updates from new data.
	"""
	# NOTE:  .__iadd__(), .copy() and I/O are forbidden, but these
	# operations will be blocked by the .numerator and .denominator
	# instances, no need to add extra code here to prevent these
	# operations
	def __init__(self, src, donor):
		self.numerator = inspiral_lr.OnlineFrankensteinLnSignalDensity.splice(src.numerator, donor.numerator)
		self.denominator = inspiral_lr.OnlineFrankensteinLnNoiseDensity.splice(src.denominator, donor.denominator)

	def finish(self):
		# no zero-lag
		self.numerator.finish()
		self.denominator.finish()
		return self


#
# =============================================================================
#
#                       False Alarm Book-Keeping Object
#
# =============================================================================
#


def binned_log_likelihood_ratio_rates_from_samples(signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples):
	"""
	Populate signal and noise BinnedLnPDF densities from a sequence of
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
	signal_lr_lnpdf_count = signal_lr_lnpdf.count
	noise_lr_lnpdf_count = noise_lr_lnpdf.count
	for ln_lamb, lnP_signal, lnP_noise in itertools.islice(samples, nsamples):
		if isnan(ln_lamb):
			raise ValueError("encountered NaN likelihood ratio")
		if isnan(lnP_signal) or isnan(lnP_noise):
			raise ValueError("encountered NaN signal or noise model probability densities")
		signal_lr_lnpdf_count[ln_lamb,] += exp(lnP_signal)
		noise_lr_lnpdf_count[ln_lamb,] += exp(lnP_noise)


#
# Class to compute ranking statistic PDFs for background-like and
# signal-like populations
#


class RankingStatPDF(object):
	ligo_lw_name_suffix = u"gstlal_inspiral_rankingstatpdf"

	@staticmethod
	def density_estimate(lnpdf, name, kernel = rate.gaussian_window(4.)):
		"""
		For internal use only.
		"""
		assert not numpy.isnan(lnpdf.array).any(), "%s log likelihood ratio PDF contain NaNs" % name
		rate.filter_array(lnpdf.array, kernel)

	@staticmethod
	def binned_log_likelihood_ratio_rates_from_samples_wrapper(queue, signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples):
		"""
		For internal use only.
		"""
		try:
			# want the forked processes to use different random
			# number sequences, so we re-seed Python and
			# numpy's random number generators here in the
			# wrapper in the hope that that takes care of it
			random.seed()
			numpy.random.seed()
			binned_log_likelihood_ratio_rates_from_samples(signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples)
			queue.put((signal_lr_lnpdf.array, noise_lr_lnpdf.array))
		except:
			queue.put((None, None))
			raise

	def __init__(self, rankingstat, signal_noise_pdfs = None, nsamples = 2**24, nthreads = 8, verbose = False):
		#
		# bailout out used by .from_xml() class method to get an
		# uninitialized instance
		#

		if rankingstat is None:
			return

		#
		# initialize binnings
		#

		self.noise_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))
		self.signal_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))
		self.zero_lag_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))
		self.segments = segmentsUtils.vote(rankingstat.segmentlists.values(), rankingstat.min_instruments)
		if rankingstat.template_ids is None:
			raise ValueError("cannot be initialized from a RankingStat that is not for a specific set of templates")
		self.template_ids = rankingstat.template_ids

		#
		# bailout used by codes that want all-zeros histograms
		#

		if not nsamples:
			return

		#
		# run importance-weighted random sampling to populate
		# binnings.
		#

		if signal_noise_pdfs is None:
			signal_noise_pdfs = rankingstat

		nthreads = int(nthreads)
		assert nthreads >= 1
		threads = []
		for i in range(nthreads):
			assert nsamples // nthreads >= 1
			q = multiprocessing.queues.SimpleQueue()
			p = multiprocessing.Process(target = lambda: self.binned_log_likelihood_ratio_rates_from_samples_wrapper(
				q,
				self.signal_lr_lnpdf,
				self.noise_lr_lnpdf,
				rankingstat.ln_lr_samples(rankingstat.denominator.random_params(), signal_noise_pdfs),
				nsamples = nsamples // nthreads
			))
			p.start()
			threads.append((p, q))
			nsamples -= nsamples // nthreads
			nthreads -= 1
			# sleep a bit to help random number seeds change
			time.sleep(1.5)
		while threads:
			p, q = threads.pop(0)
			signal_counts, noise_counts = q.get()
			self.signal_lr_lnpdf.array += signal_counts
			self.noise_lr_lnpdf.array += noise_counts
			p.join()
			if p.exitcode:
				raise Exception("sampling thread failed")
		if verbose:
			print >>sys.stderr, "done computing ranking statistic PDFs"

		#
		# apply density estimation kernels to counts
		#

		self.density_estimate(self.noise_lr_lnpdf, "noise model")
		self.density_estimate(self.signal_lr_lnpdf, "signal model")

		#
		# set the total sample count in the noise and signal
		# ranking statistic histogram equal to the total expected
		# count of the respective events from the experiment.  this
		# information is required so that when adding ranking
		# statistic PDFs in our .__iadd__() method they are
		# combined with the correct relative weights, so that
		# .__iadd__() has the effect of marginalizing the
		# distribution over the experiments being combined.
		#

		self.noise_lr_lnpdf.array *= sum(rankingstat.denominator.candidate_count_model().values()) / self.noise_lr_lnpdf.array.sum()
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.array *= rankingstat.numerator.candidate_count_model() / self.signal_lr_lnpdf.array.sum()
		self.signal_lr_lnpdf.normalize()


	def copy(self):
		new = self.__class__(None)
		new.noise_lr_lnpdf = self.noise_lr_lnpdf.copy()
		new.signal_lr_lnpdf = self.signal_lr_lnpdf.copy()
		new.zero_lag_lr_lnpdf = self.zero_lag_lr_lnpdf.copy()
		new.segments = type(self.segments)(self.segments)
		new.template_ids = self.template_ids
		return new


	def collect_zero_lag_rates(self, connection, coinc_def_id):
		# FIXME:  Gamma0 contains the template_id, switch to proper
		# column when one is available
		for ln_likelihood_ratio, template_id in connection.cursor().execute("""
SELECT
	likelihood,
	(SELECT
		Gamma0
	FROM
		sngl_inspiral
		JOIN coinc_event_map ON (
			coinc_event_map.table_name == "sngl_inspiral"
			AND coinc_event_map.event_id == sngl_inspiral.event_id
		)
	WHERE
		coinc_event_map.coinc_event_id == coinc_event.coinc_event_id
	LIMIT 1
	)
FROM
	coinc_event
WHERE
	coinc_def_id == ?
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
			if template_id not in self.template_ids:
				raise ValueError("zero-lag candidate encountered with template ID not for this RankingStatPDF")
			self.zero_lag_lr_lnpdf.count[ln_likelihood_ratio,] += 1.
		self.zero_lag_lr_lnpdf.normalize()


	def density_estimate_zero_lag_rates(self):
		# apply density estimation preserving total count, then
		# normalize PDF
		count_before = self.zero_lag_lr_lnpdf.array.sum()
		# FIXME:  should .normalize() be able to handle NaN?
		if count_before:
			self.density_estimate(self.zero_lag_lr_lnpdf, "zero lag")
			self.zero_lag_lr_lnpdf.array *= count_before / self.zero_lag_lr_lnpdf.array.sum()
		self.zero_lag_lr_lnpdf.normalize()


	def __iadd__(self, other):
		self.noise_lr_lnpdf += other.noise_lr_lnpdf
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf += other.signal_lr_lnpdf
		self.signal_lr_lnpdf.normalize()
		self.zero_lag_lr_lnpdf += other.zero_lag_lr_lnpdf
		self.zero_lag_lr_lnpdf.normalize()
		# FIXME:  now that we know what templates this is for, we
		# could conceivably impose a policy that if the segments
		# overlap then the templates must be different, and if the
		# templates are the same then the segments must be disjoint
		self.segments += other.segments
		self.template_ids |= other.template_ids
		return self


	def new_with_extinction(self):
		self = self.copy()

		# the probability that an event survives clustering is the
		# probability that no event with a higher value of the
		# ranking statistic is within +/- dt of the event in
		# question.  .noise_lr_lnpdf.count contains an accurate
		# model of the counts of pre-clustered events in each
		# ranking statistic bin, but we know the events are not
		# independent and so the total number of them cannot be
		# used to form a Poisson rate for the purpose of computing
		# the probability we desire.  it has been found,
		# empirically, that if the CCDF of pre-clustered background
		# event counts is raised to some power and the clustering
		# survival probability computed from that as though it were
		# the CCDF of a Poisson process with some effective mean
		# event rate, the result is a very good model for the
		# post-clustering distribution of ranking statistics.  we
		# have two unknown parameters:  the power to which to raise
		# the pre-clustered ranking statistic's CCDF, and the mean
		# event rate to assume.  we solve for these parameters by
		# fitting to the observed zero-lag ranking statistic
		# histogram

		x = self.noise_lr_lnpdf.bins[0].centres()
		assert (x == self.zero_lag_lr_lnpdf.bins[0].centres()).all()

		# some candidates are rejected by the ranking statistic,
		# causing there to be a spike in the zero-lag density at ln
		# L = -inf.  if enough candidates get rejected this spike
		# becomes the mode of the PDF which messes up the mask
		# constructed below for the fit.  we zero the first 40 bins
		# here to prevent that from happening (assume density
		# estimation kernel = 4 bins wide, with 10 sigma impulse
		# length)
		zl_counts = self.zero_lag_lr_lnpdf.array.copy()
		zl_counts[:40] = 0.
		if not zl_counts.any():
			raise ValueError("zero-lag counts are all zero")

		# get the noise counts
		noise_counts = self.noise_lr_lnpdf.array.copy()

		# get the zerolag counts.
		# we model the tail of the distribution - top 0.1 - 1% - where
		# clustering only effects the result at a < 1% level.
		if zl_counts.sum() < 100 * 1000:
			tail_zl_counts = zl_counts.sum() * 0.99
		else:
			tail_zl_counts = zl_counts.sum() - 300
		onepercent = zl_counts.cumsum().searchsorted(tail_zl_counts)

		# normalize the counts
		noise_counts /= noise_counts.sum()
		zl_counts /= zl_counts.sum()

		# compute survival probability
		norm = zl_counts[onepercent] / noise_counts[onepercent]
		zl_counts[onepercent:] = 0
		noise_counts[onepercent:] = 0
		survival_probability = zl_counts / noise_counts
		survival_probability[onepercent:] = norm
		survival_probability[numpy.isnan(survival_probability)] = 0.0

		# apply to background counts and signal counts
		self.noise_lr_lnpdf.array *= survival_probability
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.array *= survival_probability
		self.signal_lr_lnpdf.normalize()

		#
		# never allow PDFs that have had the extinction model
		# applied to be written to disk:  on-disk files must only
		# ever provide the original data.  forbid PDFs that have
		# been extincted from being re-extincted.
		#

		def new_with_extinction(*args, **kwargs):
			raise NotImplementedError("re-extincting an extincted RankingStatPDF object is forbidden")
		self.new_with_extinction = new_with_extinction
		def to_xml(*args, **kwargs):
			raise NotImplementedError("writing extincted RankingStatPDF object to disk is forbidden")
		self.to_xml = to_xml

		return self


	def is_healthy(self, verbose = False):
		# do we believe the PDFs are sufficiently well-defined to
		# compute FAPs and FARs?
		health = min(self.noise_lr_lnpdf.array.sum() / 1000000., self.zero_lag_lr_lnpdf.array.sum() / 10000.)
		if verbose:
			ProgressBar(text = "ranking stat. health", value = health).show()
		return health >= 1.


	@classmethod
	def get_xml_root(cls, xml, name):
		"""
		Sub-classes can use this in their overrides of the
		.from_xml() method to find the root element of the XML
		serialization.
		"""
		name = u"%s:%s" % (name, cls.ligo_lw_name_suffix)
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		return xml[0]


	@classmethod
	def from_xml(cls, xml, name):
		# find the root of the XML tree containing the
		# serialization of this object
		xml = cls.get_xml_root(xml, name)
		# create a mostly uninitialized instance
		self = cls(None)
		# populate from XML
		self.noise_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"noise_lr_lnpdf")
		self.signal_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"signal_lr_lnpdf")
		self.zero_lag_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"zero_lag_lr_lnpdf")
		self.segments = ligolw_param.get_pyvalue(xml, u"segments").strip()
		self.segments = segmentsUtils.from_range_strings(self.segments.split(",") if self.segments else [], float)
		self.template_ids = frozenset(map(int, ligolw_param.get_pyvalue(xml, u"template_ids").split(",")))
		return self

	def to_xml(self, name):
		# do not allow ourselves to be written to disk without our
		# PDFs' internal normalization metadata being up to date
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.normalize()
		self.zero_lag_lr_lnpdf.normalize()

		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.noise_lr_lnpdf.to_xml(u"noise_lr_lnpdf"))
		xml.appendChild(self.signal_lr_lnpdf.to_xml(u"signal_lr_lnpdf"))
		xml.appendChild(self.zero_lag_lr_lnpdf.to_xml(u"zero_lag_lr_lnpdf"))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"segments", ",".join(segmentsUtils.to_range_strings(self.segments))))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"template_ids", ",".join("%d" % template_id for template_id in sorted(self.template_ids))))
		return xml


#
# Class to compute false-alarm probabilities and false-alarm rates from
# ranking statistic PDFs
#


class FAPFAR(object):
	def __init__(self, rankingstatpdf):
		# input checks
		if not rankingstatpdf.zero_lag_lr_lnpdf.array.any():
			raise ValueError("RankingStatPDF's zero-lag counts are all zero")

		# save the livetime
		self.livetime = float(abs(rankingstatpdf.segments))

		# set the rate normalization LR threshold to be the mode of
		# the zero-lag LR distribution.  NOTE: this is a hack to
		# work around the extinction model's inability to model the
		# rate of extremly low significance events.  if the
		# extinction model can ever be made to model the observed
		# candidate rate at all ranking statistic values then this
		# threshold nonsense can be removed.  NOTE: some candidates
		# are rejected by the ranking statistic, causing there to
		# be a spike in the zero-lag density at ln L = -inf.  if
		# enough candidates get rejected this spike becomes the
		# mode of the PDF.  .new_with_extinction() above has a hack
		# to work around this in its noise mode fitting code, which
		# we need to reproduce here when picking the threshold
		# above which we believe the noise model to be correct
		zl = rankingstatpdf.zero_lag_lr_lnpdf.copy()
		zl.array[:40] = 0.
		rate_normalization_lr_threshold, = zl.argmax()

		# record trials factor, with safety checks
		counts = rankingstatpdf.zero_lag_lr_lnpdf.count
		assert not numpy.isnan(counts.array).any(), "zero lag log likelihood ratio counts contain NaNs"
		assert (counts.array >= 0.).all(), "zero lag log likelihood ratio rates contain negative values"
		self.count_above_threshold = counts[rate_normalization_lr_threshold:,].sum()

		# get noise model ranking stat values and event counts from
		# bins
		threshold_index = rankingstatpdf.noise_lr_lnpdf.bins[0][rate_normalization_lr_threshold]
		ranks = rankingstatpdf.noise_lr_lnpdf.bins[0].lower()[threshold_index:]
		counts = rankingstatpdf.noise_lr_lnpdf.array[threshold_index:]
		assert not numpy.isnan(counts).any(), "background log likelihood ratio rates contain NaNs"
		assert (counts >= 0.).all(), "background log likelihood ratio rates contain negative values"

		# complementary cumulative distribution function
		ccdf = counts[::-1].cumsum()[::-1]
		ccdf /= ccdf[0]

		# ccdf is P(ranking stat > threshold | a candidate).  we
		# need P(ranking stat > threshold), i.e. need to correct
		# for the possibility that no candidate is present.
		# specifically, the ccdf needs to =1-1/e at the candidate
		# identification threshold, and cdf=1/e at the candidate
		# threshold, in order for FAR(threshold) * livetime to
		# equal the actual observed number of candidates.
		ccdf = gstlalstats.poisson_p_not_0(ccdf)

		# safety checks
		assert not numpy.isnan(ranks).any(), "log likelihood ratio co-ordinates contain NaNs"
		assert not numpy.isinf(ranks).any(), "log likelihood ratio co-ordinates are not all finite"
		assert not numpy.isnan(ccdf).any(), "log likelihood ratio CCDF contains NaNs"
		assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "log likelihood ratio CCDF failed to be normalized"

		# build interpolator.
		self.ccdf_interpolator = interpolate.interp1d(ranks, ccdf)

		# record min and max ranks so we know which end of the ccdf
		# to use when we're out of bounds
		self.minrank = ranks[0]
		self.maxrank = ranks[-1]

	@gstlalstats.assert_probability
	def ccdf_from_rank(self, rank):
		return self.ccdf_interpolator(numpy.clip(rank, self.minrank, self.maxrank))

	def fap_from_rank(self, rank):
		# implements equation (8) from Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.
		return gstlalstats.fap_after_trials(self.ccdf_from_rank(rank), self.count_above_threshold)

	def rank_from_fap(self, p, tolerance = 1e-6):
		"""
		Inverts .fap_from_rank().  This function is sensitive to
		numerical noise for probabilities that are close to 1.  The
		tolerance sets the absolute error of the result.
		"""
		assert 0. <= p <= 1., "p (%g) is not a valid probability" % p
		lo, hi = self.minrank, self.maxrank
		while hi - lo > tolerance:
			mid = (hi + lo) / 2.
			mid_fap = self.fap_from_rank(mid)
			if p > mid_fap:
				# desired rank is below the middle
				hi = mid
			elif p < mid_fap:
				# desired rank is above the middle
				lo = mid
			else:
				# jackpot
				return mid
		return (hi + lo) / 2.
	rank_from_fap.__get__ = numpy.vectorize(rank_from_fap, otypes = (numpy.float64,), excluded = (0, 2, 'self', 'tolerance'))

	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv:1209.0718.  the return value is divided by T to
		# convert events/experiment to events/second.  "tdp" =
		# true-dismissal probability = 1 - single-event false-alarm
		# probability, the integral in equation (B4)
		log_tdp = numpy.log1p(-self.ccdf_from_rank(rank))
		return self.count_above_threshold * -log_tdp / self.livetime

	def rank_from_far(self, rate, tolerance = 1e-6):
		"""
		Inverts .far_from_rank() using a bisection search.  The
		tolerance sets the absolute error of the result.
		"""
		lo, hi = self.minrank, self.maxrank
		while hi - lo > tolerance:
			mid = (hi + lo) / 2.
			mid_far = self.far_from_rank(mid)
			if rate > mid_far:
				# desired rank is below the middle
				hi = mid
			elif rate < mid_far:
				# desired rank is above the middle
				lo = mid
			else:
				# jackpot
				return mid
		return (hi + lo) / 2.
	rank_from_far.__get__ = numpy.vectorize(rank_from_far, otypes = (numpy.float64,), excluded = (0, 2, 'self', 'tolerance'))

	def assign_fapfars(self, connection):
		# assign false-alarm probabilities and false-alarm rates
		# FIXME:  abusing false_alarm_rate column to store FAP,
		# move to a false_alarm_probability column??
		def as_float(f):
			def g(x):
				return float(f(x))
			return g
		connection.create_function("fap_from_rankingstat", 1, as_float(self.fap_from_rank))
		connection.create_function("far_from_rankingstat", 1, as_float(self.far_from_rank))
		connection.cursor().execute("""
UPDATE
	coinc_inspiral
SET
	false_alarm_rate = (
		SELECT
			fap_from_rankingstat(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id
	),
	combined_far = (
		SELECT
			far_from_rankingstat(coinc_event.likelihood)
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


def gen_likelihood_control_doc(xmldoc, rankingstat, rankingstatpdf):
	name = u"gstlal_inspiral_likelihood"
	node = xmldoc.childNodes[-1]
	assert node.tagName == ligolw.LIGO_LW.tagName

	if rankingstat is not None:
		node.appendChild(rankingstat.to_xml(name))

	if rankingstatpdf is not None:
		node.appendChild(rankingstatpdf.to_xml(name))

	return xmldoc


def parse_likelihood_control_doc(xmldoc):
	name = u"gstlal_inspiral_likelihood"
	try:
		rankingstat = RankingStat.from_xml(xmldoc, name)
	except ValueError:
		rankingstat = None
	try:
		rankingstatpdf = RankingStatPDF.from_xml(xmldoc, name)
	except ValueError:
		rankingstatpdf = None
	if rankingstat is None and rankingstatpdf is None:
		raise ValueError("document does not contain likelihood ratio data")
	return rankingstat, rankingstatpdf


def marginalize_pdf_urls(urls, which, ignore_missing_files = False, verbose = False):
	"""
	Implements marginalization of PDFs in ranking statistic data files.
	The marginalization is over the degree of freedom represented by
	the file collection.  One or both of the candidate parameter PDFs
	and ranking statistic PDFs can be processed, with errors thrown if
	one or more files is missing the required component.
	"""
	name = u"gstlal_inspiral_likelihood"
	data = None
	for n, url in enumerate(urls, start = 1):
		#
		# load input document
		#

		if verbose:
			print >>sys.stderr, "%d/%d:" % (n, len(urls)),
		try:
			xmldoc = ligolw_utils.load_url(url, verbose = verbose, contenthandler = RankingStat.LIGOLWContentHandler)
		except IOError:
			# IOError is raised when an on-disk file is
			# missing.  urllib2.URLError is raised when a URL
			# cannot be loaded, but this is subclassed from
			# IOError so IOError will catch those, too.
			if not ignore_missing_files:
				raise
			if verbose:
				print >>sys.stderr, "Could not load \"%s\" ... skipping as requested" % url
			continue

		#
		# extract PDF objects compute weighted sum of ranking data
		# PDFs
		#

		if which == "RankingStat":
			if data is None:
				data = RankingStat.from_xml(xmldoc, name)
			else:
				data += RankingStat.from_xml(xmldoc, name)
		elif which == "RankingStatPDF":
			if data is None:
				data = RankingStatPDF.from_xml(xmldoc, name)
			else:
				data += RankingStatPDF.from_xml(xmldoc, name)
		else:
			raise ValueError("invalid which (%s)" % which)
		xmldoc.unlink()

	return data
