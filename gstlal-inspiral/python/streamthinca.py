# Copyright (C) 2011--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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
# | Names                                       | Hash                                        | Date       | Diff to Head of Master      |
# | ------------------------------------------- | ------------------------------------------- | ---------- | --------------------------- |
# | Kipp Cannon, Chad Hanna, Jolien Creighton, Florent Robinet, B. Sathyaprakash, Duncan Meacher, T.G.G. Li | b8fef70a6bafa52e3e120a495ad0db22007caa20 | 2014-12-03 | <a href="@gstlal_inspiral_cgit_diff/python/streamthinca.py?id=HEAD&id2=b8fef70a6bafa52e3e120a495ad0db22007caa20">streamthinca.py</a> |
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


import bisect


from glue import iterutils
from glue import segments
from glue.ligolw import lsctables
from pylal import ligolw_thinca
from pylal import snglcoinc
import lal
import time


#
# =============================================================================
#
#                                Configuration
#
# =============================================================================
#


#
# allowed instrument combinations (yes, hard-coded, just take off, eh)
#


allowed_instrument_combos = frozenset([frozenset(("H1", "H2", "L1")), frozenset(("H1", "L1", "V1")), frozenset(("H1", "L1")), frozenset(("H1", "V1")), frozenset(("L1", "V1")), frozenset(("H1H2", "L1")), frozenset(("H1H2", "L1", "V1")), frozenset(("E1", "E2")), frozenset(("E1", "E3")), frozenset(("E2", "E3")), frozenset(("E1", "E2", "E3"))])


#
# =============================================================================
#
#                      pylal.ligolw_thinca Customizations
#
# =============================================================================
#


#
# sngl_inspiral<-->sngl_inspiral comparison function
#


def event_comparefunc(event_a, offset_a, event_b, offset_b, light_travel_time, delta_t):
	# NOTE:  we also require the masses and spin of the two events to
	# match, but the InspiralEventList class ensures that all event
	# pairs that make it this far are from the same template so we
	# don't need to explicitly test for that here.
	return float(abs(event_a.end + offset_a - event_b.end - offset_b)) > light_travel_time + delta_t


#
# InspiralEventList customization making use of the fact that we demand
# exact template co-incidence to increase performance.  NOTE:  the use of
# this class defeats ligolw_thinca()'s ability to apply veto segments.  We
# don't use that feature in StreamThinca so this isn't a problem for us,
# but it's something to be aware of if this gets used somewhere else.
#


class InspiralEventList(ligolw_thinca.InspiralEventList):
	@staticmethod
	def template(event):
		"""
		Returns an immutable hashable object (it can be used as a
		dictionary key) uniquely identifying the template that
		produced the given event.
		"""
		return event.mass1, event.mass2, event.spin1x, event.spin1y, event.spin1z, event.spin2x, event.spin2y, event.spin2z

	def make_index(self):
		self.index = {}
		for event in self:
			self.index.setdefault(self.template(event), []).append(event)
		for events in self.index.values():
			events.sort(key = lambda event: event.end)

	def get_coincs(self, event_a, offset_a, light_travel_time, delta_t, comparefunc):
		#
		# event_a's end time, with the time shift applied
		#

		end = event_a.end + offset_a - self.offset

		#
		# all events sharing event_a's template
		#

		try:
			events = self.index[self.template(event_a)]
		except KeyError:
			# that template didn't produce any events in this
			# instrument
			return []

		#
		# extract the subset of events from this list that pass
		# coincidence with event_a (use bisection searches for the
		# minimum and maximum allowed end times to quickly identify
		# a subset of the full list)
		#

		return [event_b for event_b in events[bisect.bisect_left(events, end - self.dt) : bisect.bisect_right(events, end + self.dt)] if not comparefunc(event_a, offset_a, event_b, self.offset, light_travel_time, delta_t)]


#
# Replace the InspiralEventList class in ligolw_thinca with ours
#


ligolw_thinca.InspiralEventList = InspiralEventList


#
# =============================================================================
#
#                                 StreamThinca
#
# =============================================================================
#


#
# on-the-fly thinca implementation
#


class StreamThinca(object):
	def __init__(self, coincidence_threshold, thinca_interval = 50.0, sngls_snr_threshold = None):
		self._xmldoc = None
		self.thinca_interval = thinca_interval	# seconds
		self.last_coincs = {}
		self.sngls_snr_threshold = sngls_snr_threshold
		self.sngl_inspiral_table = None
		self.ln_likelihood_func = None
		self.ln_likelihood_params_func = None

		# the \Delta t window not including the light travel time
		self.coincidence_threshold = coincidence_threshold

		# upper boundary of interval spanned by last invocation
		self.last_boundary = -segments.infinity()

		# set of the event ids of triggers currently in ram that
		# have already been used in coincidences
		self.event_ids = set()


	def set_coinc_params_distributions(self, coinc_params_distributions):
		self.ln_likelihood_func = snglcoinc.LnLikelihoodRatio(coinc_params_distributions)
		self.ln_likelihood_params_func = coinc_params_distributions.coinc_params
	def del_coinc_params_distributions(self):
		self.ln_likelihood_func = None
		self.ln_likelihood_params_func = None
	coinc_params_distributions = property(None, set_coinc_params_distributions, del_coinc_params_distributions, "ThincaCoincParamsDistributions instance with which to compute likelihood ratio values.")


	@property
	def max_dt(self):
		"""
		Upper bound on the time that can separate two triggers and
		they still pass coincidence, not including time shifts.
		"""
		# add 10% to coincidence window for safety + the
		# light-crossing time of the Earth
		return 1.1 * self.coincidence_threshold + 2. * lal.REARTH_SI / lal.C_SI


	def add_events(self, xmldoc, process_id, events, boundary, fapfar = None):
		# invalidate the coinc extractor in case all that follows
		# is a no-op
		self.last_coincs = {}

		# we need our own copy of the sngl_inspiral table because
		# we need a place to store a history of all the triggers,
		# and a place we can run coincidence on them.  when making
		# our copy, we can't use table.new_from_template() because
		# we need to ensure we have a Table subclass, not a DBTable
		# subclass
		if self.sngl_inspiral_table is None:
			self.sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable, lsctables.SnglInspiralTable.get_table(xmldoc).columnnames)
			# so we can watch for it changing
			assert self._xmldoc is None
			self._xmldoc = xmldoc

		# append the new row objects to our sngl_inspiral table
		self.sngl_inspiral_table.extend(events)

		# run coincidence, return non-coincident sngls.  no-op if
		# no new events
		if not events:
			return []
		return self.run_coincidence(xmldoc, process_id, boundary, fapfar = fapfar)


	def run_coincidence(self, xmldoc, process_id, boundary, fapfar = None):
		"""
		boundary = the time up-to which the trigger list can be
		assumed to be complete.
		"""
		#
		# Notes on time intervals:
		#
		#  ... -------)[----------------------)[----------------- ...
		#
		#              ^                       ^
		#              | last boundary         | boundary
		#        ^                       ^
		#        |last_bound-back_off    | boundary-back_off
		#        [----------------------)
		#             coinc segment (times of earliest single)
		#        ^
		#        | discard all singles before here
		#
		# We know all singles up-to boundary;  from boundary and on
		# the list might be incomplete.  A concidence can involve
		# triggers separated by as much as coincidence_back_off
		# (including time slide offsets).  Therefore, on this
		# iteration coincs whose earliest trigger is not later than
		# (boundary-coincidence_back_off) are complete;  coincs
		# whose earliest trigger occurs on or after
		# (boundary-coincidence_back_off) might be incomplete (we
		# might form new doubles or doubles might be promoted to
		# triples, and so on, when we fill in the singles list
		# later).
		#
		# Therefore, if for the purpose of this code we define the
		# "time" of a coinc by the time of its earliest single,
		# then on this iteration, we will construct all coincs with
		# times in [last_boundary-coincidence_back_off,
		# boundary-coincidence_back_off).  Singles that precede
		# (last_boundary-coincidence_back_off), are no longer
		# required since all coincs that can involve those triggers
		# were obtained on the last iteration.
		#

		# safety check
		assert xmldoc is self._xmldoc

		# check that we've accumulated thinca_interval seconds, and
		# that .add_events() has been called with some events since
		# the last flush
		if self.last_boundary + self.thinca_interval > boundary or self.sngl_inspiral_table is None:
			return []

		# how far apart two singles can be and still be coincident,
		# including time slide offsets.  FIXME:  assumes the
		# greatest and least offsets are in different instruments,
		# if not then this overestimates the time that can separate
		# two triggers
		time_slide_table = lsctables.TimeSlideTable.get_table(xmldoc)
		coincidence_back_off = max(offset for offset in time_slide_table.getColumnByName("offset")) - min(offset for offset in time_slide_table.getColumnByName("offset")) + self.max_dt

		# remove triggers that are too old to be useful from our
		# internal sngl_inspiral table.  save any that were never
		# used in coincidences
		discard_boundary = self.last_boundary - coincidence_back_off
		noncoinc_sngls = [row for row in self.sngl_inspiral_table if row.end < discard_boundary and row.event_id not in self.event_ids]
		iterutils.inplace_filter(lambda row: row.end >= discard_boundary, self.sngl_inspiral_table)

		# we need our own copies of these other tables because
		# sometimes ligolw_thinca wants to modify the attributes of
		# a row object after appending it to a table, which isn't
		# possible if the tables are SQL-based.  these do not store
		# any state so we create them on the fly when needed
		coinc_event_map_table = lsctables.New(lsctables.CoincMapTable)
		coinc_event_table = lsctables.New(lsctables.CoincTable)
		coinc_inspiral_table = lsctables.New(lsctables.CoincInspiralTable)

		# replace tables with our versions
		real_sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		real_coinc_event_map_table = lsctables.CoincMapTable.get_table(xmldoc)
		real_coinc_event_table = lsctables.CoincTable.get_table(xmldoc)
		real_coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(xmldoc)
		xmldoc.childNodes[-1].replaceChild(self.sngl_inspiral_table, real_sngl_inspiral_table)
		xmldoc.childNodes[-1].replaceChild(coinc_event_map_table, real_coinc_event_map_table)
		xmldoc.childNodes[-1].replaceChild(coinc_event_table, real_coinc_event_table)
		xmldoc.childNodes[-1].replaceChild(coinc_inspiral_table, real_coinc_inspiral_table)

		# synchronize our coinc_event table's ID generator with the
		# ID generator attached to the database' table object
		coinc_event_table.set_next_id(real_coinc_event_table.next_id)

		# define once-off ntuple_comparefunc() so we can pass the
		# coincidence segment in as a default value for the seg
		# keyword argument
		def ntuple_comparefunc(events, offset_vector, seg = segments.segment(self.last_boundary - coincidence_back_off, boundary - coincidence_back_off)):
			return frozenset(event.ifo for event in events) not in allowed_instrument_combos or min(event.end for event in events) not in seg

		# find coincs
		ligolw_thinca.ligolw_thinca(
			xmldoc,
			process_id = process_id,
			coinc_definer_row = ligolw_thinca.InspiralCoincDef,
			event_comparefunc = event_comparefunc,
			thresholds = self.coincidence_threshold,
			ntuple_comparefunc = ntuple_comparefunc,
			likelihood_func = self.ln_likelihood_func,
			likelihood_params_func = self.ln_likelihood_params_func,
			max_dt = self.max_dt
		)

		# assign the FAP and FAR if provided with the data to do so
		if fapfar is not None:
			coinc_event_index = dict((row.coinc_event_id, row) for row in coinc_event_table)
			gps_time_now = float(lal.UTCToGPS(time.gmtime()))
			for coinc_inspiral_row in coinc_inspiral_table:
				likelihood_ratio = coinc_event_index[coinc_inspiral_row.coinc_event_id].likelihood
				coinc_inspiral_row.combined_far = fapfar.far_from_rank(likelihood_ratio)
				# FIXME:  add a proper column to store this in
				coinc_inspiral_row.false_alarm_rate = fapfar.fap_from_rank(likelihood_ratio)

				# abuse minimum_duration column to store
				# the latency.  NOTE:  this is nonsensical
				# unless running live.
				coinc_inspiral_row.minimum_duration = gps_time_now - float(coinc_inspiral_row.end)

		# construct a coinc extractor from the XML document while
		# the tree still contains our internal table objects
		self.last_coincs = ligolw_thinca.sngl_inspiral_coincs(xmldoc)

		# synchronize the database' coinc_event table's ID
		# generator with ours
		real_coinc_event_table.set_next_id(coinc_event_table.next_id)

		# put the original table objects back
		xmldoc.childNodes[-1].replaceChild(real_sngl_inspiral_table, self.sngl_inspiral_table)
		xmldoc.childNodes[-1].replaceChild(real_coinc_event_map_table, coinc_event_map_table)
		xmldoc.childNodes[-1].replaceChild(real_coinc_event_table, coinc_event_table)
		xmldoc.childNodes[-1].replaceChild(real_coinc_inspiral_table, coinc_inspiral_table)

		# copy triggers into real output document
		if coinc_event_map_table:
			# figure out the IDs of triggers that have been
			# used in coincs for the first time, and update the
			# set of IDs of all triggers that have been used in
			# coincs
			index = dict((row.event_id, row) for row in self.sngl_inspiral_table)
			self.event_ids &= set(index)
			newids = set(coinc_event_map_table.getColumnByName("event_id")) - self.event_ids
			self.event_ids |= newids

			# find zero-lag coinc event IDs
			zero_lag_time_slide_ids = set(time_slide_id for time_slide_id, offsetvector in time_slide_table.as_dict().items() if not any(offsetvector.values()))
			zero_lag_coinc_event_ids = set(row.coinc_event_id for row in coinc_event_table if row.time_slide_id in zero_lag_time_slide_ids)

			# singles used in coincs but not in zero-lag
			# coincs.  these will be added to the
			# "non-coincident singles" list before returning
			background_coinc_sngl_ids = set(coinc_event_map_table.getColumnByName("event_id")) - set(row.event_id for row in coinc_event_map_table if row.coinc_event_id in zero_lag_coinc_event_ids)
			background_coinc_sngls = map(index.__getitem__, background_coinc_sngl_ids)

			# copy rows into target tables.
			for event_id in newids:
				real_sngl_inspiral_table.append(index[event_id])
			map(real_coinc_event_map_table.append, coinc_event_map_table)
			map(real_coinc_event_table.append, coinc_event_table)
			map(real_coinc_inspiral_table.append, coinc_inspiral_table)
		else:
			background_coinc_sngls = []

		# save all sngls above the requested sngls SNR threshold
		# (all snlgs that participated in coincs are already in the
		# document, so only need to check for ones in the
		# non-coincident sngls list for this iteration)
		if self.sngls_snr_threshold is not None:
			for event in noncoinc_sngls:
				if event.snr >= self.sngls_snr_threshold:
					real_sngl_inspiral_table.append(event)

		# record boundary
		self.last_boundary = boundary

		# return sngls that are not coincident in zero-lag
		return noncoinc_sngls + background_coinc_sngls


	def flush(self, xmldoc, process_id, fapfar = None):
		# invalidate the coinc extractor in case run_coincidence()
		# is a no-op.
		self.last_coincs = {}

		# coincidence.  don't bother unless .add_events() has been
		# called since the last flush()
		if self._xmldoc is not None:
			noncoinc_sngls = self.run_coincidence(xmldoc, process_id, segments.infinity(), fapfar = fapfar)
		else:
			noncoinc_sngls = []

		# any event that hasn't been used in a coinc by now will
		# never be
		if self.sngl_inspiral_table is not None:
			noncoinc_sngls.extend(row for row in self.sngl_inspiral_table if row.event_id not in self.event_ids)
			self.sngl_inspiral_table.unlink()
			self.sngl_inspiral_table = None
		self.event_ids.clear()

		# last_boundary must be reset to -infinity so that it looks
		# like a fresh copy of the stream thinca instance
		self.last_boundary = -segments.infinity()

		# it's now safe to work with a different document
		self._xmldoc = None

		# return non-coincident sngls
		return noncoinc_sngls
