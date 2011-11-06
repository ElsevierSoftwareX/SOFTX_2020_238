# Copyright (C) 2011  Kipp Cannon, Chad Hanna, Drew Keppel
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


from glue import iterutils
from glue import segments
from glue.ligolw import lsctables
from pylal import ligolw_thinca
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
lsctables.LIGOTimeGPS = LIGOTimeGPS


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#


#
# on-the-fly thinca implementation
#


class StreamThinca(object):
	def __init__(self, dataobj, coincidence_threshold, coincidence_back_off, thinca_interval = 50.0):
		self.dataobj = dataobj
		self.process_table = lsctables.New(lsctables.ProcessTable)
		self.process_params_table = lsctables.New(lsctables.ProcessParamsTable)
		self.sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable, dataobj.sngl_inspiral_table.columnnames)
		self.coinc_event_map_table = lsctables.New(lsctables.CoincMapTable)
		self.last_boundary = -segments.infinity()
		# when using the normal coincidence function from
		# ligolw_thinca this is the e-thinca parameter.  when using
		# a \Delta t only coincidence test it's the \Delta t window
		# not including the light travel time
		self.coincidence_threshold = coincidence_threshold
		self.coincidence_back_off = coincidence_back_off + max(abs(offset) for offset in dataobj.time_slide_table.getColumnByName("offset"))
		self.thinca_interval = thinca_interval
		# set of the event ids of triggers currently in ram that
		# have already been used in coincidences
		self.ids = set()

	def run_coincidence(self, boundary):
		# wait until we've accumulated thinca_interval seconds
		if self.last_boundary + self.thinca_interval > boundary:
			return

		# remove triggers that are too old to be useful
		discard_boundary = self.last_boundary - self.coincidence_back_off
		iterutils.inplace_filter(lambda row: row.get_end() >= discard_boundary, self.sngl_inspiral_table)

		# replace tables with our versions
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.process_table, self.dataobj.process_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.process_params_table, self.dataobj.process_params_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.coinc_event_map_table, self.dataobj.coinc_event_map_table)

		# find coincs.  gstlal_inspiral's triggers cause a
		# divide-by-zero error in the effective SNR function used
		# for lalapps_inspiral triggers, so we replace it with one
		# that works for the duration of the ligolw_thinca() call.
		def event_comparefunc(event_a, offset_a, event_b, offset_b, light_travel_time, delta_t):
			return (event_a.mass1 != event_b.mass1) or (event_a.mass2 != event_b.mass2) or (event_a.chi != event_b.chi) or (abs(event_a.get_end() + offset_a - event_b.get_end() - offset_b) > light_travel_time + delta_t)
		def ntuple_comparefunc(events, offset_vector, seg = segments.segment(self.last_boundary, boundary)):
			return set(event.ifo for event in events) not in (set(("H1", "H2", "L1")), set(("H1", "L1", "V1")), set(("H1", "L1")), set(("H1", "V1")), set(("L1", "V1"))) or ligolw_thinca.coinc_inspiral_end_time(events, offset_vector) not in seg
		def get_effective_snr(self, fac):
			return self.snr
		orig_get_effective_snr, ligolw_thinca.SnglInspiral.get_effective_snr = ligolw_thinca.SnglInspiral.get_effective_snr, get_effective_snr
		ligolw_thinca.ligolw_thinca(
			self.dataobj.xmldoc,
			process_id = self.dataobj.process.process_id,
			EventListType = ligolw_thinca.InspiralEventList,
			CoincTables = ligolw_thinca.InspiralCoincTables,
			coinc_definer_row = ligolw_thinca.InspiralCoincDef,
			event_comparefunc = event_comparefunc,
			thresholds = self.coincidence_threshold,
			ntuple_comparefunc = ntuple_comparefunc
		)
		ligolw_thinca.SnglInspiral.get_effective_snr = orig_get_effective_snr

		# put the original table objects back
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.process_table, self.process_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.process_params_table, self.process_params_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.coinc_event_map_table, self.coinc_event_map_table)
		del self.process_table[:]
		del self.process_params_table[:]

		# record boundary
		self.last_boundary = boundary

	def appsink_new_buffer(self, elem):
		# replace the sngl_inspiral table with our version.  in
		# addition to replacing the table object in the xml tree,
		# we also need to replace the attribute in the dataobj
		# because that's what appsink_new_buffer() will write to
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.sngl_inspiral_table, self.dataobj.sngl_inspiral_table)
		orig_sngl_inspiral_table, self.dataobj.sngl_inspiral_table = self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table

		# chain to normal function in pipeparts.  after this, the
		# new triggers will have been appended to our
		# sngl_inspiral_table
		prev_len = len(self.sngl_inspiral_table)
		self.dataobj.appsink_new_buffer(elem)

		# convert the new row objects to the type required by
		# ligolw_thinca()
		for i in range(prev_len, len(self.sngl_inspiral_table)):
			old = self.sngl_inspiral_table[i]
			self.sngl_inspiral_table[i] = new = ligolw_thinca.SnglInspiral()
			for col in self.sngl_inspiral_table.columnnames:
				setattr(new, col, getattr(old, col))

		# coincidence
		if self.sngl_inspiral_table:
			# since the triggers are appended to the table, we
			# can rely on the last one to provide an estimate
			# of the most recent time stamps to come out of the
			# pipeline
			self.run_coincidence(self.sngl_inspiral_table[-1].get_end() - self.coincidence_back_off)

		# put the original sngl_inspiral table back
		self.dataobj.sngl_inspiral_table = orig_sngl_inspiral_table
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table)

		# copy triggers into real output document
		if self.coinc_event_map_table:
			index = dict((row.event_id, row) for row in self.sngl_inspiral_table)
			self.ids &= set(index)
			newids = set(self.coinc_event_map_table.getColumnByName("event_id")) - self.ids
			self.ids |= newids
			self.coinc_event_map_table.reverse()	# so the loop that follows preserves order
			while self.coinc_event_map_table:
				self.dataobj.coinc_event_map_table.append(self.coinc_event_map_table.pop())
			for id in newids:
				self.dataobj.sngl_inspiral_table.append(index[id])
			if self.dataobj.connection is not None:
				self.dataobj.connection.commit()

	def flush(self):
		# replace the sngl_inspiral table with our version
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.sngl_inspiral_table, self.dataobj.sngl_inspiral_table)

		# coincidence
		self.run_coincidence(segments.infinity())

		# put the original sngl_inspiral table back
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table)

		# copy triggers into real output document
		newids = set(self.coinc_event_map_table.getColumnByName("event_id")) - self.ids
		self.ids |= newids
		while self.coinc_event_map_table:
			self.dataobj.coinc_event_map_table.append(self.coinc_event_map_table.pop())
		while self.sngl_inspiral_table:
			row = self.sngl_inspiral_table.pop()
			if row.event_id in newids:
				self.dataobj.sngl_inspiral_table.append(row)
