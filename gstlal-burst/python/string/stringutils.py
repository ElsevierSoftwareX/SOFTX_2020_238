# Copyright (C) 2009--2018  Kipp Cannon
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


from __future__ import print_function


try:
	from fpconst import NegInf
except ImportError:
	NegInf = float("-inf")
import itertools
import math
import numpy
from scipy import interpolate
import sys


from ligo.lw import ligolw
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import lsctables
from ligo.lw import utils as ligolw_utils
from ligo import segments
from ligo.segments import utils as segmentsUtils
from lalburst import snglcoinc
from . import string_lr_far

from lal import rate
from gstlal import stats as gstlalstats

__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
from .git_version import date as __date__
from .git_version import version as __version__


#
# =============================================================================
#
#                             Likelihood Machinery
#
# =============================================================================
#


#
# Parameter distributions
#


class RankingStat(snglcoinc.LnLikelihoodRatioMixin):
	ligo_lw_name_suffix = u"string_rankingstat"

	@ligolw_array.use_in
	@ligolw_param.use_in
	@lsctables.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass

	def __init__(self, delta_t, snr_threshold, num_templates, instruments = frozenset(("H1", "L1", "V1")), min_instruments = 2):
		self.numerator = string_lr_far.LnSignalDensity(instruments = instruments, delta_t = delta_t, snr_threshold = snr_threshold, num_templates = num_templates, min_instruments = min_instruments)
		self.denominator = string_lr_far.LnNoiseDensity(instruments = instruments, delta_t = delta_t, snr_threshold = snr_threshold, num_templates = num_templates, min_instruments = min_instruments)
		self.candidates = string_lr_far.LnLRDensity(instruments = instruments, delta_t = delta_t, snr_threshold = snr_threshold, num_templates = num_templates, min_instruments = min_instruments)

	@property
	def instruments(self):
		return self.denominator.instruments

	@property
	def min_instruments(self):
		return self.denominator.min_instruments
	
	def __iadd__(self, other):
		self.numerator += other.numerator
		self.denominator += other.denominator
		self.candidates += other.candidates
		return self

	def __call__(self, **kwargs):
		# NOTE now we use the default definition, but the
		# ranking statistic definition can be customized here
		# (to include e.g. veto cuts, chisq cuts, ...)
		return super(RankingStat, self).__call__(**kwargs)

	def copy(self):
		new = type(self)(
			instruments = self.instruments,
			min_instruments = self.min_instruments,
			num_templates = self.num_templates,
			delta_t = self.delta_t,
			snr_threshold = self.snr_threshold
		)
		new.numerator = self.numerator.copy()
		new.denominator = self.denominator.copy()
		new.candidates = self.candidates.copy()
		return new

	def kwargs_from_triggers(self, events, offsetvector):
		assert len(events) >= self.min_instruments

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
		ref_start, ref_offset = reference.start_time, offsetvector[reference.ifo]
		# segment spanned by reference event
		seg = segments.segment(ref_start, ref_start + reference.duration)
		# initially populate segs dictionary shifting reference
		# instrument's segment according to offset vectors
		segs = dict((instrument, seg.shift(ref_offset - offsetvector[instrument])) for instrument in self.instruments)
		
		# for any any real triggers we have, use their true
		# intervals
		segs.update((event.ifo, segments.segment(event.start_time, event.start_time+event.duration)) for event in events)

		return dict(
			segments = segs, 
			snr2s = dict((event.ifo, event.snr**2.) for event in events),
			chi2s_over_snr2s = dict((event.ifo, event.chisq / event.chisq_dof / event.snr**2.) for event in events),
			durations = dict((event.ifo, event.duration) for event in events)
		)

	def ln_lr_from_triggers(self, events, offsetvector):
		return self(**self.kwargs_from_triggers(events, offsetvector))

	def finish(self):
		self.numerator.finish()
		self.denominator.finish()
		self.candidates.finish()
		return self

	@classmethod
	def get_xml_root(cls, xml, name):
		name = u"%s:%s" % (name, cls.ligo_lw_name_suffix)
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		return xml[0]

	@classmethod
	def from_xml(cls, xml, name = u"string_cusp"):
		xml = cls.get_xml_root(xml, name)
		self = cls.__new__(cls)
		self.numerator = string_lr_far.LnSignalDensity.from_xml(xml, "numerator")
		self.denominator = string_lr_far.LnNoiseDensity.from_xml(xml, "denominator")
		self.candidates = string_lr_far.LnLRDensity.from_xml(xml, "candidates")
		return self

	def to_xml(self, name = u"string_cusp"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.numerator.to_xml("numerator"))
		xml.appendChild(self.denominator.to_xml("denominator"))
		xml.appendChild(self.candidates.to_xml("candidates"))
		return xml


#
# =============================================================================
#
#                             Ranking Statistic PDF
#
# =============================================================================
#


def binned_log_likelihood_ratio_rates_from_samples(signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples):
	"""
	Populate signal and noise BinnedLnPDF densities from a sequence of
	samples (which can be a generator). The first nsamples elements
	from the sequence are used. The samples must be a sequence of
	three-element tuples (or sequences) in which the first element is a
	value of the ranking statistic (likelihood ratio) and the second
	and third elements are the logs of the probabilities of obtaining
	that value of the ranking statistic in the signal and noise populations
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

	return signal_lr_lnpdf.array, noise_lr_lnpdf.array


#
# Class to compute ranking statistic PDFs for background-like and
# signal-like populations
#


class RankingStatPDF(object):
	ligo_lw_name_suffix = u"string_rankingstatpdf"

	@staticmethod
	def density_estimate(lnpdf, name, kernel = rate.gaussian_window(4.)):
		"""
		For internal use only.
		"""
		assert not numpy.isnan(lnpdf.array).any(), "%s log likelihood ratio PDF contain NaNs" % name
		rate.filter_array(lnpdf.array, kernel)

	def __init__(self, rankingstat, nsamples = 2**24, verbose = False):
		#
		# bailout used by .from_xml() class method to get an
		# uninitialized instance
		#

		if rankingstat is None:
			return

		#
		# initialize binnings
		#

		self.noise_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(-10., 30., 3000),)))
		self.signal_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(-10., 30., 3000),)))
		self.candidates_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(-10., 30., 3000),)))

		#
		# obtain analyzed segments that will be used to obtain livetime
		#

		self.segments = segmentsUtils.vote(rankingstat.denominator.triggerrates.segmentlistdict().values(),rankingstat.min_instruments)

		#
		# run importance-weighted random sampling to populate binnings.
		#

		self.signal_lr_lnpdf.array, self.noise_lr_lnpdf.array = binned_log_likelihood_ratio_rates_from_samples(
			self.signal_lr_lnpdf,
			self.noise_lr_lnpdf,
			rankingstat.ln_lr_samples(rankingstat.denominator.random_params(), rankingstat),
			nsamples = nsamples)

		if verbose:
			print("done computing ranking statistic PDFs", file=sys.stderr) 

		#
		# apply density estimation kernels to counts
		#

		self.density_estimate(self.noise_lr_lnpdf, "noise model")
		self.density_estimate(self.signal_lr_lnpdf, "signal model")

		#
		# set the total sample count in the noise and signal
		# ranking statistic histogram equal to the total expected
		# count of the respective events from the experiment. This
		# information is required so that when adding ranking
		# statistic PDFs in our .__iadd__() method they are
		# combined with the correct relative weights, so that
		# .__iadd__() has the effect of marginalizing the
		# distribution over the experients being combined.
		#

		self.noise_lr_lnpdf.array /= self.noise_lr_lnpdf.array.sum()
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.array /= self.signal_lr_lnpdf.array.sum()
		self.signal_lr_lnpdf.normalize()


	def copy(self):
		new = self.__class__(None)
		new.noise_lr_lnpdf = self.noise_lr_lnpdf.copy()
		new.signal_lr_lnpdf = self.signal_lr_lnpdf.copy()
		new.candidates_lr_lnpdf = self.candidates_lr_lnpdf.copy()
		new.segments = type(self.segments)(self.segments)
		return new

	def collect_zero_lag_rates(self, connection, coinc_def_id):
		for ln_likelihood_ratio_tuple in connection.cursor().execute("""
SELECT
	likelihood
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
			# FIXME don't know sqlite syntax
			ln_likelihood_ratio = ln_likelihood_ratio_tuple[0]
			assert ln_likelihood_ratio is not None, "null likelihood ratio encountered.  probably coincs have not been ranked"
			self.candidates_lr_lnpdf.count[ln_likelihood_ratio,] += 1.
		self.candidates_lr_lnpdf.normalize()

	
	def density_estimate_zero_lag_rates(self):
		# apply density estimation preserving total count, then
		# normalize PDF
		count_before = self.candidates_lr_lnpdf.array.sum()
		# FIXME: should .normalize() be able to handle NaN?
		if count_before:
			self.density_estimate(self.candidates_lr_lnpdf, "zero lag")
			self.candidates_lr_lnpdf.array *= count_before / self.candidates_lr_lnpdf.array.sum()
		self.candidates_lr_lnpdf.normalize()
	

	def __iadd__(self, other):
		self.noise_lr_lnpdf += other.noise_lr_lnpdf
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf += other.signal_lr_lnpdf
		self.signal_lr_lnpdf.normalize()
		self.candidates_lr_lnpdf += other.candidates_lr_lnpdf
		self.candidates_lr_lnpdf.normalize()
		self.segments += other.segments
		return self

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
	def from_xml(cls, xml, name = u"string_cusp"):
		# find the root of the XML tree containing the
		# serialization of this object
		xml = cls.get_xml_root(xml, name)
		# create a mostly uninitialized instance
		self = cls(None)
		# popuate from XML
		self.noise_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"noise_lr_lnpdf")
		self.signal_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"signal_lr_lnpdf")
		self.candidates_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"candidates_lr_lnpdf")
		self.segments = ligolw_param.get_pyvalue(xml, u"segments").strip()
		self.segments = segmentsUtils.from_range_strings(self.segments.split(",") if self.segments else [], float)
		return self
	
	def to_xml(self, name = u"string_cusp"):
		# do not allow ourselves to be written to disk without our
		# PDF's internal normalization metadata being up to date
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.normalize()
		self.candidates_lr_lnpdf.normalize()

		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.noise_lr_lnpdf.to_xml(u"noise_lr_lnpdf"))
		xml.appendChild(self.signal_lr_lnpdf.to_xml(u"signal_lr_lnpdf"))
		xml.appendChild(self.candidates_lr_lnpdf.to_xml(u"candidates_lr_lnpdf"))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"segments", ",".join(segmentsUtils.to_range_strings(self.segments))))
		return xml


#
# =============================================================================
#
#                       False alarm rates/probabilities 
#
# =============================================================================
#


#
# Class to compute false-alarm probabilities and false-alarm rates from
# ranking statistic PDFs
#

class FAPFAR(object):
	def __init__(self, rankingstatpdf):
		# input checks
		if not rankingstatpdf.candidates_lr_lnpdf.array.any():
			raise ValueError("RankingStatPDF's zero-lag counts are all zero")

		# obtain livetime from rankingstatpdf
		self.livetime = float(abs(rankingstatpdf.segments)) 
		
		# set the rate normalization LR threshold to the mode of
		# the zero-lag LR distribution.
		zl = rankingstatpdf.candidates_lr_lnpdf.copy()
		rate_normalization_lr_threshold, = zl.argmax()
		print("lr_threshold %f" % rate_normalization_lr_threshold, file=sys.stderr)

		# record trials factor, with safety checks
		counts = rankingstatpdf.candidates_lr_lnpdf.count
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

		# ccdf is P(ranking statistic > threshold | a candidate).
		# we need P(rankins statistic > threshold), i.e. need to
		# correct for the possibility that no candidate is present.
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
		# arXiv: 1209.0718
		return gstlalstats.fap_after_trials(self.ccdf_from_rank(rank), self.count_above_threshold)
	
	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv: 1209.0718. the return value is divided by T to
		# convert events/experiment to events/second. "tdp" =
		# true-dismissal probability = 1 - single-event FAP, the
		# integral in equation (B4)
		log_tdp = numpy.log1p(-self.ccdf_from_rank(rank))
		return self.count_above_threshold * -log_tdp / self.livetime
	
	def assign_fapfars(self, connection):
		# assign false-alarm probabilities and false-alarm rates
		def as_float(f):
			def g(x):
				return float(f(x))
			return g
		connection.create_function("fap_from_rankingstat", 1, as_float(self.fap_from_rank))
		connection.create_function("far_from_rankingstat", 1, as_float(self.far_from_rank))
		connection.cursor().execute("""
UPDATE
	coinc_burst
SET
	false_alarm_probability = (
		SELECT
			fap_from_rankingstat(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_burst.coinc_event_id
	),
	false_alarm_rate = (
		SELECT
			far_from_rankingstat(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_burst.coinc_event_id
	)
""")


#
# =============================================================================
#
#                                     I/O
#
# =============================================================================
#


def marginalize_rankingstat(filenames, verbose = False):
	rankingstat = None
	for n, filename in enumerate(filenames, 1):
		if verbose:
			print("%d/%d:" % (n, len(filenames)), end=' ', file=sys.stderr)
		xmldoc = ligolw_utils.load_filename(filename, verbose = verbose, contenthandler = RankingStat.LIGOLWContentHandler)
		if rankingstat is None:
			rankingstat = RankingStat.from_xml(xmldoc)
		else:
			rankingstat += RankingStat.from_xml(xmldoc)
		xmldoc.unlink()
	return rankingstat


def marginalize_rankingstatpdf(filenames, verbose = False):
	rankingstatpdf = None
	for n, filename in enumerate(filenames, 1):
		if verbose:
			print("%d/%d:" % (n, len(filenames)), end=' ', file=sys.stderr)
		xmldoc = ligolw_utils.load_filename(filename, verbose = verbose, contenthandler = RankingStat.LIGOLWContentHandler)
		if rankingstatpdf is None:
			rankingstatpdf = RankingStatPDF.from_xml(xmldoc)
		else:
			rankingstatpdf += RankingStatPDF.from_xml(xmldoc)
		xmldoc.unlink()
	return rankingstatpdf


#
# =============================================================================
#
#                              Database Utilities
#
# =============================================================================
#


def create_recovered_ln_likelihood_ratio_table(connection, coinc_def_id):
	"""
	Create a temporary table named "recovered_ln_likelihood_ratio"
	containing two columns:  "simulation_id", the simulation_id of an
	injection, and "ln_likelihood_ratio", the highest log likelihood
	ratio at which that injection was recovered by a coincidence of
	type coinc_def_id.
	"""
	cursor = connection.cursor()
	cursor.execute("""
CREATE TEMPORARY TABLE recovered_ln_likelihood_ratio (simulation_id TEXT PRIMARY KEY, ln_likelihood_ratio REAL)
	""")
	cursor.execute("""
INSERT OR REPLACE INTO
	recovered_ln_likelihood_ratio
SELECT
	sim_burst.simulation_id AS simulation_id,
	MAX(coinc_event.likelihood) AS ln_likelihood_ratio
FROM
	sim_burst
	JOIN coinc_event_map AS a ON (
		a.table_name == "sim_burst"
		AND a.event_id == sim_burst.simulation_id
	)
	JOIN coinc_event_map AS b ON (
		b.coinc_event_id == a.coinc_event_id
	)
	JOIN coinc_event ON (
		b.table_name == "coinc_event"
		AND b.event_id == coinc_event.coinc_event_id
	)
WHERE
	coinc_event.coinc_def_id == ?
GROUP BY
	sim_burst.simulation_id
	""", (coinc_def_id,))
	cursor.close()


def create_sim_burst_best_string_sngl_map(connection, coinc_def_id):
	"""
	Construct a sim_burst --> best matching sngl_burst mapping.
	"""
	connection.cursor().execute("""
CREATE TEMPORARY TABLE
	sim_burst_best_string_sngl_map
AS
	SELECT
		sim_burst.simulation_id AS simulation_id,
		(
			SELECT
				sngl_burst.event_id
			FROM
				coinc_event_map AS a
				JOIN coinc_event_map AS b ON (
					b.coinc_event_id == a.coinc_event_id
				)
				JOIN coinc_event ON (
					coinc_event.coinc_event_id == a.coinc_event_id
				)
				JOIN sngl_burst ON (
					b.table_name == 'sngl_burst'
					AND b.event_id == sngl_burst.event_id
				)
			WHERE
				a.table_name == 'sim_burst'
				AND a.event_id == sim_burst.simulation_id
				AND coinc_event.coinc_def_id == ?
			ORDER BY
				(sngl_burst.chisq / sngl_burst.chisq_dof) / (sngl_burst.snr * sngl_burst.snr)
			LIMIT 1
		) AS event_id
	FROM
		sim_burst
	WHERE
		event_id IS NOT NULL
	""", (coinc_def_id,))
