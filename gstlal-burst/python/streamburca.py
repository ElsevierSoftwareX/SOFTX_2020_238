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


#
# =============================================================================
#
#                                 StreamBurca
#
# =============================================================================
#


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


	def push(self, instrument, events, t_complete):
		"""
		Push new triggers into the coinc engine.  Returns True if
		the coinc engine's internal state has changed in a way that
		might enable new candidates to be constructed, False if
		not.
		"""
		return self.time_slide_graph.push(instrument, events, t_complete)


	def pull(self, coinc_sieve = None, flush = False):
		#
		# iterate over coincidences
		#

		newly_reported = []
		for node, events in self.time_slide_graph.pull(newly_reported = newly_reported, coinc_sieve = coinc_sieve, flush = flush):
			# construct row objects for coinc tables

			coinc, coincmaps, multiburst = self.coinc_tables.coinc_rows(self.process_id, node.time_slide_id, events, u"sngl_burst")

			# finally, append coinc to tables

			self.coinc_tables.append_coinc(coinc, coincmaps, multiburst)

		# add any triggers that have been used in coincidences for
		# the first time to the sngl_inspiral table

		self.sngl_burst_table.extend(newly_reported)
