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
Read triggers from a ligolw file and output a stream of
application/x-lal-snglinspiral.
"""
__author__ = "Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"

import bisect

from gstlal.pipeutil import *
from gst.extend.pygobject import gproperty
from pylal.xlal.datatypes.snglinspiraltable import SnglInspiralTable
from glue import iterutils
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.segments import segment


def trigger_time(trig):
	return trig.end_time * gst.SECOND + trig.end_time_ns

class lal_ligolwtriggersrc(gst.BaseSrc):
	__gstdetails__ = (
		"LIGO_LW trigger source",
		"Source",
		__doc__,
		__author__
	)
	gproperty(
		gobject.TYPE_STRING,
		"xml-location",
		"Name of LIGO Light Weight XML file containing list of templates",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_STRING,
		"sqlite-location",
		"Name of LIGO Light Weight SQLite file containing list of templates",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"buffer-duration",
		"Duration of each buffer (nanoseconds)",
		1, gst.CLOCK_TIME_NONE, 4 * gst.SECOND,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"start-time",
		"Time from which to start playback (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, gst.CLOCK_TIME_NONE,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"duration",
		"Duration for which to produce triggers (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, gst.CLOCK_TIME_NONE,
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
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = (int) 1
			""")
		),
	)


	def __init__(self):
		super(lal_ligolwtriggersrc, self).__init__()
		self.set_do_timestamp(False)
		self.set_format(gst.FORMAT_TIME)
		self.src_pads().next().use_fixed_caps()
		for prop in self.props:
			self.set_property(prop.name, prop.default_value)


	def do_start(self):
		"""GstBaseSrc->start virtual method"""

		xml_location = self.get_property("xml-location")
		tmp_space = self.get_property("tmp-space")
		sqlite_location = self.get_property("sqlite-location")
		if not (xml_location is None) ^ (sqlite_location is None):
			self.error("must set xml_location or sqlite_location")
			return False
		if (sqlite_location is not None) and (tmp_space is None):
			self.error("if sqlite_location is provided, must provide tmp_space")
			return False

		start_time = self.get_property("start-time")
		duration = self.get_property("duration")
		end_time = start_time + duration
		requested_seg = segment(start_time, end_time)

		# override SnglInspiralTable to create rows of type SnglInspiralTable
		lsctables.SnglInspiralTable.RowType = SnglInspiralTable

		# XML
		if xml_location is not None:
			doc = utils.load_filename(xml_location, gz=xml_location.endswith(".gz"))
		else:
			try:
					import sqlite3
			except ImportError:
					# pre 2.5.x
					from pysqlite2 import dbapi2 as sqlite3
			from glue.ligolw import dbtables
			working_filename = dbtables.get_connection_filename(sqlite_location, tmp_path=tmp_space, verbose=True)
			connection = sqlite3.connect(working_filename)
			dbtables.DBTable_set_connection(connection)
			doc = dbtables.get_xml(connection)

		# FIXME: extract segment info
		if start_time == gst.CLOCK_TIME_NONE or end_time == gst.CLOCK_TIME_NONE:
			#seglistdict = llwapp.segmentlistdict_fromsearchsummary(doc)
			raise NotImplementedError

		# read triggers
		trigs = table.get_table(doc, lsctables.SnglInspiralTable.tableName)

		# pack times and triggers together as a list of tuples
		self.__time_trig_tuples = [(trigger_time(trig), trig) for trig in trigs if trigger_time(trig) in requested_seg]
		self.__time_trig_tuples.sort()

		self.__last_time = start_time
		self.__stream_end_time = end_time
		self.__ntriggers = 0

		return True


	def do_stop(self):
		"""GstBaseSrc->stop virtual method"""
		self.__time_trig_tuples = None
		self.__ntriggers = 0
		return True


	def do_check_get_range(self):
		"""GstBaseSrc->check_get_range virtual method"""
		return True


	def do_is_seekable(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return False


	def do_create(self, offset, size):
		"""GstBaseSrc->create virtual method"""

		buffer_duration = self.get_property("buffer-duration")
		timestamp = self.__last_time

		if timestamp >= self.__stream_end_time:
			gst.info('timestamp %d is greater than stream end time %d, sending EOS' % (timestamp, self.__stream_end_time))
			self.src_pads().next().push_event(gst.event_new_eos())
			return (gst.FLOW_UNEXPECTED, None)

		end_time = timestamp + buffer_duration
		if end_time > self.__stream_end_time:
			end_time = self.__stream_end_time
			buffer_duration = end_time - timestamp

		# Select triggers
		start_ind = bisect.bisect_left(self.__time_trig_tuples, (timestamp,))
		stop_ind = bisect.bisect_right(self.__time_trig_tuples, (end_time,))
		num_trigs = stop_ind - start_ind

		# create output buffer
		rowsize = len(buffer(SnglInspiralTable()))
		pad = self.src_pads().next()
		(retval, buf) = pad.alloc_buffer(0, rowsize * num_trigs, pad.get_property("caps")) # TODO set offset

		if retval != gst.FLOW_OK:
			return (retval, None)

		# copy triggers into buffer
		trigs = (trig for (time, trig) in self.__time_trig_tuples[start_ind:stop_ind])
		for i, trig in enumerate(trigs):
			buf[i*rowsize:(i+1)*rowsize] = buffer(trig)

		# set metadata
		buf.timestamp = timestamp
		buf.duration = buffer_duration
		buf.offset = self.__ntriggers
		self.__ntriggers += num_trigs
		buf.offset_end = self.__ntriggers

		# FIXME: add gap info

		self.__last_time = end_time
		return (gst.FLOW_OK, buf)


# Register element class
gstlal_element_register(lal_ligolwtriggersrc)
