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
# STATUS: reviewed with actions
#
# | Names                                       | Hash                                        | Date        | Diff to Head of Master      |
# | ------------------------------------------- | ------------------------------------------- | ----------  | --------------------------- |
# | Kipp Cannon, Chad Hanna, Jolien Creighton, Florent Robinet, B. Sathyaprakash, Duncan Meacher, T.G.G. Li | b8fef70a6bafa52e3e120a495ad0db22007caa20 | 2014-12-03 | <a href="@gstlal_inspiral_cgit_diff/python/streamthinca.py?id=HEAD&id2=b8fef70a6bafa52e3e120a495ad0db22007caa20">streamthinca.py</a> |
# | Kipp Cannon, Chad Hanna, Jolien Creighton, B. Sathyaprakash, Duncan Meacher                             | 72875f5cb241e8d297cd9b3f9fe309a6cfe3f716 | 2015-11-06 | <a href="@gstlal_inspiral_cgit_diff/python/streamthinca.py?id=HEAD&id2=72875f5cb241e8d297cd9b3f9fe309a6cfe3f716">streamthinca.py</a> |
#
# #### Action items
#
# - Question: Is it possible for the offline pipeline to begin producing tiggers after a certain time rather than waiting for all the inspiral jobs to get over? Will be particularly useful if the data length is ~ months or ~ year. Should also avoid producing massive amount of data, right?
# - L300+: Please document within the code that the FAR column is used to store FAP so that future developers don't get confused what that column represents


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import time


from ligo.lw import ligolw
from ligo.lw import lsctables
import lal
from lalburst import snglcoinc
from lalinspiral import thinca
from ligo import segments
from ligo.segments import utils as segmentsUtils


#
# =============================================================================
#
#                              last_coincs helper
#
# =============================================================================
#


#
# the last_coincs machine, which is used to construct documents for upload
# to gracedb, used to be implemented by a class in thinca.py which was
# originally written for this purpose.  when the coincidence engine
# switched to a native streaming design, that older implementation was no
# longer suitable because it indexes the entire document.  we have had to
# replace that implementation with this, here.  the old one is now unused,
# but it's potentially useful so it has been left behind and this one put
# here instead.
#


class last_coincs(object):
	def __init__(self, xmldoc):
		#
		# find all tables
		#

		self.process_table = lsctables.ProcessTable.get_table(xmldoc)
		self.process_params_table = lsctables.ProcessParamsTable.get_table(xmldoc)
		self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		self.coinc_def_table = lsctables.CoincDefTable.get_table(xmldoc)
		self.coinc_event_table = lsctables.CoincTable.get_table(xmldoc)
		self.coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(xmldoc)
		self.coinc_event_map_table = lsctables.CoincMapTable.get_table(xmldoc)
		self.time_slide_table = lsctables.TimeSlideTable.get_table(xmldoc)

		#
		# index the process, process params, and time_slide tables
		#

		self.process_index = dict((row.process_id, row) for row in self.process_table)
		self.process_params_index = {}
		for row in self.process_params_table:
			self.process_params_index.setdefault(row.process_id, []).append(row)
		self.time_slide_index = {}
		for row in self.time_slide_table:
			self.time_slide_index.setdefault(row.time_slide_id, []).append(row)

		#
		# find the sngl_inspiral<-->sngl_inspiral coinc_definer
		#

		self.coinc_def, = (row for row in self.coinc_def_table if row.search == thinca.InspiralCoincDef.search and row.search_coinc_type == thinca.InspiralCoincDef.search_coinc_type)

		#
		# coinc event metadata
		#

		self.sngl_inspiral_index = {}
		self.coinc_event_index = {}
		self.coinc_event_maps_index = {}
		self.coinc_inspiral_index = {}


	def add(self, events, coinc, coincmaps, coinc_inspiral):
		# FIXME:  these checks might be over-doing it.  I just
		# don't want any surprises, but maybe when we are confident
		# the code works they should be removed
		assert coinc.coinc_def_id == self.coinc_def.coinc_def_id
		assert coinc.coinc_event_id == coinc_inspiral.coinc_event_id
		assert coinc.process_id in self.process_index
		assert coinc.time_slide_id in self.time_slide_index
		assert all(event.process_id == coinc.process_id for event in events)
		assert all(coinc_event_map.coinc_event_id == coinc.coinc_event_id for coinc_event_map in coincmaps)
		assert set(event.event_id for event in events) == set(coinc_event_map.event_id for coinc_event_map in coincmaps)
		self.sngl_inspiral_index[coinc.coinc_event_id] = events
		self.coinc_event_index[coinc.coinc_event_id] = coinc
		self.coinc_event_maps_index[coinc.coinc_event_id] = coincmaps
		self.coinc_inspiral_index[coinc.coinc_event_id] = coinc_inspiral


	def clear(self):
		self.sngl_inspiral_index.clear()
		self.coinc_event_index.clear()
		self.coinc_event_maps_index.clear()
		self.coinc_inspiral_index.clear()


	def sngl_inspirals(self, coinc_event_id):
		return self.sngl_inspiral_index[coinc_event_id]


	def __iter__(self):
		return iter(self.coinc_event_index)


	def __nonzero__(self):
		return bool(self.coinc_event_index)


	def __getitem__(self, coinc_event_id):
		newxmldoc = ligolw.Document()
		ligolw_elem = newxmldoc.appendChild(ligolw.LIGO_LW())

		# when making these, we can't use .copy() method of Table
		# instances because we need to ensure we have a Table
		# subclass, not a DBTable subclass
		new_process_table = ligolw_elem.appendChild(lsctables.New(lsctables.ProcessTable, self.process_table.columnnamesreal))
		new_process_params_table = ligolw_elem.appendChild(lsctables.New(lsctables.ProcessParamsTable, self.process_params_table.columnnamesreal))
		new_sngl_inspiral_table = ligolw_elem.appendChild(lsctables.New(lsctables.SnglInspiralTable, self.sngl_inspiral_table.columnnamesreal))
		new_coinc_def_table = ligolw_elem.appendChild(lsctables.New(lsctables.CoincDefTable, self.coinc_def_table.columnnamesreal))
		new_coinc_event_table = ligolw_elem.appendChild(lsctables.New(lsctables.CoincTable, self.coinc_event_table.columnnamesreal))
		new_coinc_inspiral_table = ligolw_elem.appendChild(lsctables.New(lsctables.CoincInspiralTable, self.coinc_inspiral_table.columnnamesreal))
		new_coinc_event_map_table = ligolw_elem.appendChild(lsctables.New(lsctables.CoincMapTable, self.coinc_event_map_table.columnnamesreal))
		new_time_slide_table = ligolw_elem.appendChild(lsctables.New(lsctables.TimeSlideTable, self.time_slide_table.columnnamesreal))

		new_coinc_def_table.append(self.coinc_def)
		coincevent = self.coinc_event_index[coinc_event_id]
		new_time_slide_table.extend(self.time_slide_index[coincevent.time_slide_id])

		new_sngl_inspiral_table.extend(self.sngl_inspiral_index[coinc_event_id])
		new_coinc_event_table.append(coincevent)
		new_coinc_event_map_table.extend(self.coinc_event_maps_index[coinc_event_id])
		new_coinc_inspiral_table.append(self.coinc_inspiral_index[coinc_event_id])

		for process_id in set(new_sngl_inspiral_table.getColumnByName("process_id")) | set(new_coinc_event_table.getColumnByName("process_id")) | set(new_time_slide_table.getColumnByName("process_id")):
			# process row is required
			new_process_table.append(self.process_index[process_id])
			try:
				new_process_params_table.extend(self.process_params_index[process_id])
			except KeyError:
				# process_params rows are optional
				pass

		return newxmldoc


#
# =============================================================================
#
#                                 StreamThinca
#
# =============================================================================
#


class backgroundcollector(object):
	def __init__(self):
		self.zerolag_singles = set()
		self.timeshifted_coincs = set()

	def push(self, event_ids, offset_vector):
		if any(offset_vector.values()):
			if len(event_ids) > 1:
				self.timeshifted_coincs.update(event_ids)
		elif len(event_ids) == 1:
			self.zerolag_singles.update(event_ids)

	def pull(self, snr_min, hl_on, flushed_events):
		index = dict((id(event), event) for event in flushed_events)
		flushed_ids = set(index)
		background_ids = self.timeshifted_coincs & flushed_ids
		self.timeshifted_coincs -= flushed_ids
		# put all virgo and kagra in their own background
		background_ids |= set(event_id for event_id in self.zerolag_singles & flushed_ids if ((float(index[event_id].end) in hl_on) or (index[event_id].ifo not in ("H1", "L1"))))
		self.zerolag_singles -= flushed_ids
		return [event for event in map(index.__getitem__, background_ids) if event.snr >= snr_min]


class StreamThinca(object):
	def __init__(self, xmldoc, process_id, delta_t, min_instruments = 2, sngls_snr_threshold = None):
		self.ln_lr_from_triggers = None
		self.delta_t = delta_t
		self.min_instruments = min_instruments
		self.sngls_snr_threshold = sngls_snr_threshold
		self.set_xmldoc(xmldoc, process_id)
		self.clustered_sngl_ids = set()


	def set_xmldoc(self, xmldoc, process_id):
		self.coinc_tables = thinca.InspiralCoincTables(xmldoc, thinca.InspiralCoincDef)
		self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		self.process_params_table = lsctables.ProcessParamsTable.get_table(xmldoc)
		self.last_coincs = last_coincs(xmldoc)
		self.process_id = process_id
		self.time_slide_graph = snglcoinc.TimeSlideGraph(
			thinca.coincgen_doubles,
			lsctables.TimeSlideTable.get_table(xmldoc).as_dict(),
			self.delta_t,
			min_instruments = self.min_instruments
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


	def pull(self, rankingstat, fapfar = None, zerolag_rankingstatpdf = None, coinc_sieve = None, flush = False, cluster = False, cap_singles = False, FAR_trialsfactor = 1.0, template_id_time_map = None):
		# NOTE:  rankingstat is not used to compute the ranking
		# statistic, it supplies the detector livetime segment
		# lists to determine which triggers are eligible for
		# inclusion in the background model and is the destination
		# for triggers identified for inclusion in the background
		# model. self.ln_lr_from_triggers is the ranking statistic
		# function (if set).

		# extract times when instruments were producing SNR.  used
		# to define "on instruments" for coinc tables, as a safety
		# check for impossible triggers, and to identify triggers
		# suitable for use in defining the background PDFs.  will
		# only need segment information for the times for which the
		# queues will yield triggers, so use a bisection search to
		# clip the lists to reduce subsequent operation count.

		age = float(self.time_slide_graph.age)
		snr_segments = segments.segmentlistdict((instrument, ratebinlist[ratebinlist.value_slice_to_index(slice(age, None))].segmentlist()) for instrument, ratebinlist in rankingstat.denominator.triggerrates.items())

		#
		# iterate over coincidences
		#

		gps_time_now = float(lal.UTCToGPS(time.gmtime()))
		newly_reported = []
		flushed = []
		flushed_unused = []
		self.last_coincs.clear()
		max_last_coinc_snr = {}
		for node, events in self.time_slide_graph.pull(newly_reported = newly_reported, flushed = flushed, flushed_unused = flushed_unused, coinc_sieve = coinc_sieve, event_collector = self.backgroundcollector, flush = flush):
			# construct row objects for coinc tables.

			coinc, coincmaps, coinc_inspiral = self.coinc_tables.coinc_rows(self.process_id, node.time_slide_id, events, seglists = snr_segments)

			# some tasks for zero-lag candidates

			if node.is_zero_lag:
				# populate ranking statistic's zero-lag
				# PDFs with triggers from all zero-lag
				# candidates

				for event in events:
					rankingstat.zerolag.increment(event)

			# latency goes in minimum_duration column.  NOTE:
			# latency is nonsense unless running live.  FIXME:
			# add a proper column for latency

			coinc_inspiral.minimum_duration = gps_time_now - float(coinc_inspiral.end)

			# finally, append coinc to tables

			if cluster:
				max_last_coinc_snr.setdefault(node, None)
				if max_last_coinc_snr[node] is None or coinc_inspiral.snr > max_last_coinc_snr[node][3].snr:
					max_last_coinc_snr[node] = (events, coinc, coincmaps, coinc_inspiral)
			else:
				self.coinc_tables.append_coinc(coinc, coincmaps, coinc_inspiral)

				# add events to the zero-lag ranking
				# statistic histogram

				if zerolag_rankingstatpdf is not None and coinc.likelihood is not None:
					zerolag_rankingstatpdf.zero_lag_lr_lnpdf.count[coinc.likelihood,] += 1

				self.last_coincs.add(events, coinc, coincmaps, coinc_inspiral)


		for node in max_last_coinc_snr:
			if max_last_coinc_snr[node] is not None:
				events, coinc, coincmaps, coinc_inspiral = max_last_coinc_snr[node]
				# assign ranking statistic, FAP and FAR
				if self.ln_lr_from_triggers is not None:
					coinc.likelihood = self.ln_lr_from_triggers(events, node.offset_vector)
					if fapfar is not None:
						# FIXME:  add proper columns to
						# store these values in
						coinc_inspiral.combined_far = fapfar.far_from_rank(coinc.likelihood) * FAR_trialsfactor
						if len(events) == 1 and cap_singles and coinc_inspiral.combined_far < 1. / fapfar.livetime:
							coinc_inspiral.combined_far = 1. / fapfar.livetime	
						coinc_inspiral.false_alarm_rate = fapfar.fap_from_rank(coinc.likelihood)
				if zerolag_rankingstatpdf is not None and coinc.likelihood is not None:
					zerolag_rankingstatpdf.zero_lag_lr_lnpdf.count[coinc.likelihood,] += 1

				self.coinc_tables.append_coinc(coinc, coincmaps, coinc_inspiral)
				self.last_coincs.add(events, coinc, coincmaps, coinc_inspiral)
				self.sngl_inspiral_table.extend([sngl_trigger for sngl_trigger in events if sngl_trigger.event_id not in self.clustered_sngl_ids])
				self.clustered_sngl_ids |= set(e.event_id for e in events)
				if template_id_time_map is not None:
					# The same template should have the same offset regardless of ifo, so just take the first one
					offset = [template_id_time_map[int(sngl_trigger.Gamma0)] for sngl_trigger in events][0]
					row = [row for row in self.process_params_table if row.param == u'--upload-time-before-merger'][0]
					row.value=str(offset)

		# add selected singles to the noise model

		if flushed:
			# times when H and L were generating
			# SNR.  used to select zero-lag singles for
			# inclusion in the denominator.

			if "H1" in snr_segments and "L1" in snr_segments:
				hl_on = snr_segments["H1"] & snr_segments["L1"]
			else:
				hl_on = segments.segmentlist([])
			# FIXME:  this is needed to work around rounding
			# problems in safety checks below, trying to
			# compare GPS trigger times to float segment
			# boundaries (the boundaries don't have enough
			# precision to know if triggers near the edge are
			# in or out).  it would be better not to have to
			# screw around like this.
			hl_on.protract(1e-3)  # 1 ms

			for event in self.backgroundcollector.pull(rankingstat.snr_min, hl_on, flushed):
				rankingstat.denominator.increment(event)

		# add any triggers that have been used in coincidences for
		# the first time to the sngl_inspiral table
		# FIXME:  because this information comes from the
		# coincidence code, which is not aware of the clustering,
		# we record a lot of singles that aren't really used for
		# any (retained) coincs.

		if not cluster:
			self.sngl_inspiral_table.extend(newly_reported)

		# save all sngls above the requested sngls SNR threshold.
		# all sngls that participated in coincs are already in the
		# document, so only need to check for ones being flushed
		# and that were never used.

		if self.sngls_snr_threshold is not None:
			self.sngl_inspiral_table.extend(event for event in flushed_unused if event.snr >= self.sngls_snr_threshold)

		# return the triggers that have been flushed
		return flushed
