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
import scipy.stats
import sys


from ligo.lw import ligolw
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import lsctables
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process
import lal
from lal import rate
from ligo.segments import utils as segmentsUtils
from lalburst import offsetvector
from lalburst import snglcoinc

from gstlal import string_lr_far


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
# Make a look-up table of time-of-arrival triangulators
#


def triangulators(timing_uncertainties):
	"""
	Return a dictionary of snglcoinc.TOATriangulator objects
	initialized for a variety of instrument combinations.
	timing_uncertainties is a dictionary of instrument->$\\Delta t$
	pairs.  The return value is a dictionary of (instrument
	tuple)->TOATrangulator mappings.  The instrument names in each
	tuple are sorted in alphabetical order, and the triangulators are
	constructed with the instruments in that order (the the
	documentation for snglcoinc.TOATriangulator for more information).

	Example:

	>>> x = triangulators({"H1": 0.005, "L1": 0.005, "V1": 0.005})

	constructs a dictionary of triangulators for every combination of
	two or more instruments that can be constructed from those three.

	The program lalapps_string_plot_binj can be used to measure the
	timing uncertainties for the instruments in a search.
	"""
	allinstruments = sorted(timing_uncertainties.keys())

	triangulators = {}
	for n in range(2, len(allinstruments) + 1):
		for instruments in itertools.combinations(allinstruments, n):
			triangulators[instruments] = snglcoinc.TOATriangulator([lal.cached_detector_by_prefix[instrument].location for instrument in instruments], [timing_uncertainties[instrument] for instrument in instruments])

	return triangulators


#
# A binning for instrument combinations
#
# FIXME:  we decided that the coherent and null stream naming convention
# would look like
#
# H1H2:LSC-STRAIN_HPLUS, H1H2:LSC-STRAIN_HNULL
#
# and so on.  i.e., the +, x and null streams from a coherent network would
# be different channels from a single instrument whose name would be the
# mash-up of the names of the instruments in the network.  that is
# inconsisntent with the "H1H2+", "H1H2-" shown here, so this needs to be
# fixed but I don't know how.  maybe it'll go away before it needs to be
# fixed.
#


class InstrumentBins(rate.HashableBins):
	"""
	Example:

	>>> x = InstrumentBins()
	>>> x[frozenset(("H1", "L1"))]
	55
	>>> x.centres()[55]
	frozenset(['H1', 'L1'])
	"""

	names = ("E0", "E1", "E2", "E3", "G1", "H1", "H2", "H1H2+", "H1H2-", "L1", "V1")

	def __init__(self, names):
		super(InstrumentBins, self).__init__(frozenset(combo) for n in range(len(names) + 1) for combo in itertools.combinations(names, n))

	# FIXME:  hack to allow instrument binnings to be included as a
	# dimension in multi-dimensional PDFs by defining a volume for
	# them.  investigate more sensible ways to do this.  maybe NDBins
	# and BinnedDensity should understand the difference between
	# functional and parametric co-ordinates.
	def lower(self):
		return numpy.arange(0, len(self), dtype = "double")
	def upper(self):
		return numpy.arange(1, len(self) + 1, dtype = "double")

	xml_bins_name = u"instrumentbins"

# NOTE:  side effect of importing this module:
rate.NDBins.xml_bins_name_mapping.update({
	InstrumentBins.xml_bins_name: InstrumentBins,
	InstrumentBins: InstrumentBins.xml_bins_name
})


#
# for ranking statistic
#

def kwarggeniter(d, min_instruments):
	d = tuple(sorted(d.items()))
	return map(dict, itertools.chain(*(itertools.combinations(d, i) for i in range(min_instruments, len(d) + 1))))

def kwarggen(snrs, chi2s_over_snr2s, min_instruments):
	for snrs, chi2s_over_snr2s in zip(
		kwarggeniter(snrs, min_instruments),
		kwarggeniter(chi2s_over_snr2s, min_instruments)
	):
		yield {
			"snrs": snrs,
			"chi2s_over_snr2s": chi2s_over_snr2s,
		}


#
# Parameter distributions
#


class StringCoincParamsDistributions(snglcoinc.LnLikelihoodRatioMixin):
	ligo_lw_name_suffix = u"stringcusp_coincparamsdistributions"

	@ligolw_array.use_in
	@ligolw_param.use_in
	@lsctables.use_in
	class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
		pass

	def __init__(self, instruments, snr_threshold, min_instruments=2):
		self.triangulators = triangulators(dict.fromkeys(instruments, 8e-5))
		self.numerator = string_lr_far.LnSignalDensity(instruments = instruments, min_instruments = min_instruments, snr_threshold = snr_threshold)
		self.denominator = string_lr_far.LnNoiseDensity(instruments = instruments, min_instruments = min_instruments, snr_threshold = snr_threshold)
		self.candidates = string_lr_far.LnLRDensity(instruments = instruments, min_instruments = min_instruments, snr_threshold = snr_threshold)

	@property
	def min_instruments(self):
		return self.denominator.min_instruments
	
	def __call__(self, **kwargs):
		"""
		Evaluate the ranking statistic.
		"""
		# Full lnL ranking stat, defined to be the largest lnL from
		# all allowed subsets of trigges. Maximizes over 2+ IFO combos.
		return max(super(StringCoincParamsDistribution, self).__call__(**kwargs) for kwargs in kwarggen(min_instruments = self.min_instruments, **kwargs))

	def __iadd__(self, other):
		if type(self) != type(other):
			raise TypeError(other)
		if set(self.triangulators.keys()) != set(other.triangulators.keys()):
			raise ValueError("incompatible instruments")
		self.numerator += other.numerator
		self.denominator += other.denominator
		self.candidates += other.candidates
		return self

	def copy(self):
		new = type(self)([])
		new.triangulators = self.triangulators	# share reference
		new.numerator = self.numerator.copy()
		new.denominator = self.denominator.copy()
		new.candidates = self.candidates.copy()
		return new

	def coinc_params(self, events, offsetvector):
		params = {}

		#
		# check for coincs that have been vetoed entirely
		#

		if len(events) < 2:
			return params

		#
		# Initialize the parameter dictionary, sort the events by
		# instrument name (the multi-instrument parameters are defined for
		# the instruments in this order and the triangulators are
		# constructed this way too), and retrieve the sorted instrument
		# names
		#

		events = tuple(sorted(events, key = lambda event: event.ifo))
		instruments = tuple(event.ifo for event in events)

		return dict(
			snrs = dict((event.ifo, event.snr) for event in events),
			chi2s_over_snr2s = dict((event.ifo, event.chisq / event.chisq_dof / event.snr**2.) for event in events),
		)

	def ln_lr_from_triggers(self, events, offsetvector):
		return self(**self.coinc_params(events, offsetvector))

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
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = cls([])
		self.numerator = string_lr_far.LnSignalDensity.from_xml(xml, "numerator")
		self.denominator = string_lr_far.LnNoiseDensity.from_xml(xml, "denominator")
		self.candidates = string_lr_far.LnLRDensity.from_xml(xml, "candidates")
		instruments = self.candidates.instruments
		self.triangulators = triangulators(dict.fromkeys(instruments, 8e-5))
		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.numerator.to_xml("numerator"))
		xml.appendChild(self.denominator.to_xml("denominator"))
		xml.appendChild(self.candidates.to_xml("candidates"))
		return xml


#
# I/O
#


def load_likelihood_data(filenames, verbose = False):
	coinc_params = None
	seglists = None
	for n, filename in enumerate(filenames, 1):
		if verbose:
			print("%d/%d:" % (n, len(filenames)), end=' ', file=sys.stderr)
		xmldoc = ligolw_utils.load_filename(filename, verbose = verbose, contenthandler = StringCoincParamsDistributions.LIGOLWContentHandler)
		this_coinc_params = StringCoincParamsDistributions.from_xml(xmldoc, u"string_cusp_likelihood")
		this_seglists = lsctables.SearchSummaryTable.get_table(xmldoc).get_out_segmentlistdict(lsctables.ProcessTable.get_table(xmldoc).get_ids_by_program(u"lalapps_string_meas_likelihood")).coalesce()
		xmldoc.unlink()
		if coinc_params is None:
			coinc_params = this_coinc_params
		else:
			coinc_params += this_coinc_params
		if seglists is None:
			seglists = this_seglists
		else:
			seglists |= this_seglists
	return coinc_params, seglists


#
# =============================================================================
#
#                                   Livetime
#
# =============================================================================
#


def time_slides_livetime(seglists, time_slides, min_instruments, verbose = False, clip = None):
	"""
	seglists is a segmentlistdict of times when each of a set of
	instruments were on, time_slides is a sequence of
	instrument-->offset dictionaries, each vector of offsets in the
	sequence is applied to the segmentlists and the total time during
	which at least min_instruments were on is summed and returned.  If
	clip is not None, after each offset vector is applied to seglists
	the result is intersected with clip before computing the livetime.
	If verbose is True then progress reports are printed to stderr.
	"""
	livetime = 0.0
	seglists = seglists.copy()	# don't modify original
	N = len(time_slides)
	if verbose:
		print("computing the live time for %d time slides:" % N, file=sys.stderr)
	for n, time_slide in enumerate(time_slides):
		if verbose:
			print("\t%.1f%%\r" % (100.0 * n / N), end=' ', file=sys.stderr)
		seglists.offsets.update(time_slide)
		if clip is None:
			livetime += float(abs(segmentsUtils.vote(seglists.values(), min_instruments)))
		else:
			livetime += float(abs(segmentsUtils.vote((seglists & clip).values(), min_instruments)))
	if verbose:
		print("\t100.0%", file=sys.stderr)
	return livetime


def time_slides_livetime_for_instrument_combo(seglists, time_slides, instruments, verbose = False, clip = None):
	"""
	like time_slides_livetime() except computes the time for which
	exactly the instruments given by the sequence instruments were on
	(and nothing else).
	"""
	livetime = 0.0
	# segments for instruments that must be on
	onseglists = seglists.copy(keys = instruments)
	# segments for instruments that must be off
	offseglists = seglists.copy(keys = set(seglists) - set(instruments))
	N = len(time_slides)
	if verbose:
		print("computing the live time for %s in %d time slides:" % (", ".join(instruments), N), file=sys.stderr)
	for n, time_slide in enumerate(time_slides):
		if verbose:
			print("\t%.1f%%\r" % (100.0 * n / N), end=' ', file=sys.stderr)
		onseglists.offsets.update(time_slide)
		offseglists.offsets.update(time_slide)
		if clip is None:
			livetime += float(abs(onseglists.intersection(onseglists.keys()) - offseglists.union(offseglists.keys())))
		else:
			livetime += float(abs((onseglists & clip).intersection(onseglists.keys()) - offseglists.union(offseglists.keys())))
	if verbose:
		print("\t100.0%", file=sys.stderr)
	return livetime


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
	Construct a sim_burst --> best matching coinc_event mapping.
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


def create_sim_burst_best_string_coinc_map(connection, coinc_def_id):
	"""
	Construct a sim_burst --> best matching coinc_event mapping for
	string cusp injections and coincs.
	"""
	# FIXME:  this hasn't finished being ported from the inspiral code
	connection.cursor().execute("""
CREATE TEMPORARY TABLE
	sim_burst_best_string_coinc_map
AS
	SELECT
		sim_burst.simulation_id AS simulation_id,
		(
			SELECT
				coinc_inspiral.coinc_event_id
			FROM
				coinc_event_map AS a
				JOIN coinc_event_map AS b ON (
					b.coinc_event_id == a.coinc_event_id
				)
				JOIN coinc_inspiral ON (
					b.table_name == 'coinc_event'
					AND b.event_id == coinc_inspiral.coinc_event_id
				)
			WHERE
				a.table_name == 'sim_burst'
				AND a.event_id == sim_burst.simulation_id
				AND coinc_event.coinc_def_id == ?
			ORDER BY
				(sngl_burst.chisq / sngl_burst.chisq_dof) / (sngl_burst.snr * sngl_burst.snr)
			LIMIT 1
		) AS coinc_event_id
	FROM
		sim_burst
	WHERE
		coinc_event_id IS NOT NULL
	""", (coinc_def_id,))
