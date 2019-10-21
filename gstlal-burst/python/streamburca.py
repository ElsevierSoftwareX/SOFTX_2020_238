# Copyright (C) 2011--2019  Kipp Cannon, Chad Hanna, Drew Keppel
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
# The python module to implement streaming coincidence
#
# ### Review Status
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from ligo.lw import lsctables
from lalburst import snglcoinc
from lalburst import burca
from ligo import segments
from ligo.segments import utils as segmentsUtils


#
# =============================================================================
#
#                                 StreamBurca
#
# =============================================================================
#


class backgroundcollector(object):
	def __init__(self):
		self.zerolag_singles = set()
		self.timeshifted_coincs = set()

	# Used at snglcoinc
	def push(self, event_ids, offset_vector):
		# time shifted data?
		if any(offset_vector.values()):
			if len(event_ids) > 1:
				self.timeshifted_coincs.update(event_ids)
		elif len(event_ids) == 1:
			self.zerolag_singles.update(event_ids)

	def pull(self, two_or_more_instruments, flushed_events):
		index = dict((id(event), event) for event in flushed_events)
		flushed_ids = set(index)
		background_ids = self.timeshifted_coincs & flushed_ids
		self.timeshifted_coincs -= flushed_ids
		background_ids |= set(event_id for event_id in self.zerolag_singles & flushed_ids if float(index[event_id].peak_time) in two_or_more_instruments)
		self.zerolag_singles -= flushed_ids
		return [event for event in map(index.__getitem__, background_ids)]


class StreamBurca(object):
	def __init__(self, xmldoc, process_id, delta_t, min_instruments = 2, verbose = False):
		self.delta_t = delta_t
		self.min_instruments = min_instruments
		self.verbose = verbose
		self.set_xmldoc(xmldoc, process_id)


	def set_xmldoc(self, xmldoc, process_id):
		self.coinc_tables = burca.StringCuspCoincTables(xmldoc, burca.StringCuspBBCoincDef)
		self.sngl_burst_table = lsctables.SnglBurstTable.get_table(xmldoc)
		self.process_id = process_id
		self.time_slide_graph = snglcoinc.TimeSlideGraph(
			burca.string_coincgen_doubles,
			lsctables.TimeSlideTable.get_table(xmldoc).as_dict(),
			self.delta_t,
			min_instruments = self.min_instruments,
			verbose = self.verbose
		)
		self.backgroundcollector = backgroundcollector()


	def push(self, instrument, events, t_complete):
		"""
		Push new triggers into the coinc engine.  Returns True if
		the coinc engine's internal state has changed in a way that
		might enable new candidates to be constructed, False if
		not.
		"""
		return self.time_slide_graph.push(instrument, events, t_complete)


	def pull(self, rankingstat, snr_segments, coinc_sieve = None, flush = False):
		#
		# iterate over coincidences
		#

		newly_reported = []
		flushed = []
		flushed_unused = []
		for node, events in self.time_slide_graph.pull(newly_reported = newly_reported, flushed = flushed, flushed_unused = flushed_unused, coinc_sieve = coinc_sieve, event_collector = self.backgroundcollector, flush = flush, verbose = False):
			# for exact template match
			if not burca.StringCuspCoincTables.ntuple_comparefunc(events, node.offset_vector):
				# construct row objects for coinc tables

				coinc, coincmaps, coinc_burst = self.coinc_tables.coinc_rows(self.process_id, node.time_slide_id, events, u"sngl_burst")

				# finally, append coinc to tables

				self.coinc_tables.append_coinc(coinc, coincmaps, coinc_burst)

		# add singles into the noise model
		if flushed:
			# times when at least 2 instruments were generating SNR.
			# Used to select zero-lag singles for inclusion in the
			# denominator.
			two_or_more_instruments = segmentsUtils.vote(snr_segments.values(), 2)

			for event in self.backgroundcollector.pull(two_or_more_instruments, flushed):
				rankingstat.denominator.increment(event)

		# add any triggers that have been used in coincidences for
		# the first time to the sngl_burst table

		self.sngl_burst_table.extend(newly_reported)
