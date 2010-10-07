# Copyright (C) 2010  Nickolas Fotopoulos
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
"""
Read triggers from astream of application/x-lal-snglinspiral and write out a
ligolw XML or SQLite file to disk.
"""
__author__ = "Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"

import bisect
import operator

import numpy as np

from gstlal.pipeutil import *
from gstlal import pipeio
from gst.extend.pygobject import gproperty
from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils

# overrides
from pylal.xlal.datatypes.snglinspiraltable import SnglInspiralTable
lsctables.SnglInspiralTable.RowType = SnglInspiralTable
lsctables.SnglInspiralTable.next_id = lsctables.SnglInspiralID(1)
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
lsctables.LIGOTimeGPS = LIGOTimeGPS

class lal_ligolwtriggersink(gst.BaseSink):
	__gstdetails__ = (
		"LIGO_LW trigger sink",
		"Sink",
		__doc__,
		__author__
	)
	gproperty(
		gobject.TYPE_STRING,
		"xml-location",
		"Name of LIGO Light Weight XML file containing single-detector triggers",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_STRING,
		"sqlite-location",
		"Name of LIGO Light Weight SQLite file containing single-detector triggers",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_STRING,
		"tmp-space",
		"Directory in which to temporarily store work",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = (int) [1, MAX]
			""")
		),
	)

	def __init__(self):
		super(lal_ligolwtriggersink, self).__init__()
		for prop in self.props:
			if prop.name in ("xml-location"):
				self.set_property(prop.name, prop.default_value)

	def do_start(self):
		xml_location = self.get_property("xml-location")
		if xml_location is None:
			self.error("Require xml-location argument")
			return False

		# create new document
		self.xmldoc = ligolw.Document()

		# Append the LIGO_LW tag.
		self.xmldoc.appendChild(ligolw.LIGO_LW())

		# Add a SnglInspiralTable.
		self.sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable)
		self.xmldoc.childNodes[0].appendChild(self.sngl_inspiral_table)

		# Add CoincEventTable.
		self.coinc_event_table = lsctables.New(lsctables.CoincTable)
		self.xmldoc.childNodes[0].appendChild(self.coinc_event_table)

		# Add CoincDefinerTable.
		self.coinc_definer_table = lsctables.New(lsctables.CoincDefTable)
		self.xmldoc.childNodes[0].appendChild(self.coinc_definer_table)
		coinc_definer = self.coinc_definer_table.RowType(coinc_def_id = self.coinc_definer_table.get_next_id(), search = u"inspiral", search_coinc_type = 0, description = u"sngl_inspiral<-->sngl_inspiral coincidences")
		self.coinc_definer_table.append(coinc_definer)

		# Add a CoincInspiralTable.
		self.coinc_inspiral_table = lsctables.New(lsctables.CoincInspiralTable)
		self.xmldoc.childNodes[0].appendChild(self.coinc_inspiral_table)

		# Add a CoincEventMapTable.
		self.coinc_event_map_table = lsctables.New(lsctables.CoincMapTable)
		self.xmldoc.childNodes[0].appendChild(self.coinc_event_map_table)

		return True

	def do_stop(self):
		xml_location = self.get_property("xml-location")
		utils.write_filename(self.xmldoc, xml_location, gz=xml_location.endswith(".gz"))
		return True

	def do_render(self, inbuf):
		self.warning("timestamp, duration, len = %d, %d, %d" % (inbuf.timestamp, inbuf.duration, inbuf.offset))
		for sngls in pipeio.sngl_inspiral_groups_from_buffer(inbuf):
			# Fill in rows of tables.
			coinc_event = self.coinc_event_table.RowType()
			coinc_event.coinc_event_id = self.coinc_event_table.get_next_id()
			coinc_event.instruments = None
			coinc_event.nevents = len(sngls)
			coinc_event.process_id = sngls[0].process_id
			coinc_event.time_slide_id = None
			coinc_event.likelihood = None
			coinc_event.coinc_def_id = self.coinc_definer_table[0].coinc_def_id
			self.coinc_event_table.append(coinc_event)

			coinc_inspiral = self.coinc_inspiral_table.RowType()
			coinc_inspiral.coinc_event_id = coinc_event.coinc_event_id
			coinc_inspiral.mchirp = sngls[0].mchirp
			coinc_inspiral.mass = sngls[0].mtotal
			coinc_inspiral.false_alarm_rate = None
			coinc_inspiral.combined_far = reduce(operator.mul, (s.alpha for s in sngls), 1.)
			mean_end_time = sum([LIGOTimeGPS(sngl.end_time, sngl.end_time_ns) for sngl in sngls]) / len(sngls)
			coinc_inspiral.end_time = mean_end_time.seconds
			coinc_inspiral.end_time_ns = mean_end_time.nanoseconds
			coinc_inspiral.ifos = None
			coinc_inspiral.snr = np.sqrt(sum([sngl.snr**2 for sngl in sngls]))
			self.coinc_inspiral_table.append(coinc_inspiral)

			for i, sngl in enumerate(sngls):
				sngl.event_id = self.sngl_inspiral_table.get_next_id()
				self.sngl_inspiral_table.append(sngl)

				coinc_event_map = self.coinc_event_map_table.RowType()
				coinc_event_map.coinc_event_id = coinc_event.coinc_event_id
				coinc_event_map.table_name = lsctables.SnglInspiralTable.tableName
				coinc_event_map.event_id = sngl.event_id
				self.coinc_event_map_table.append(coinc_event_map)

		# Done.
		return gst.FLOW_OK

# Register element class
gstlal_element_register(lal_ligolwtriggersink)
