# Copyright (C) 2010 Erin Kara
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
Send Inspiral coincidences to GraCeDB.  See https://archie.phys.uwm.edu/gracedb/
"""
__author__ = "Erin Kara <erin.kara@ligo.org>, Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gstlal import pipeio

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS

import sys
import subprocess
from StringIO import StringIO
import numpy


class lal_gracedbsink(gst.BaseSink):
	__gstdetails__ = (
		'GraceDB Sink Element',
		'Sink',
		__doc__,
		__author__
	)

	__gsttemplates__ = gst.PadTemplate(
		"sink",
		gst.PAD_SINK,
		gst.PAD_ALWAYS,
		gst.caps_from_string("""
			application/x-lal-snglinspiral,
			channels = (int) [2, MAX]
		""")
	)

	__gproperties__ = {
		'group': (
			gobject.TYPE_STRING,
			'Group name',
			'Group name (one of CBC, Burst, Stochastic, CW, Test)',
			'Test', # Default value
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'type': (
			gobject.TYPE_STRING,
			'Type',
			'Event type (one of Q, LowMass, MBTAOnline, HighMass, X, GRB, Injection, CWB, Omega, Ringdown)',
			'LowMass', # Default value
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
	}

	def do_set_property(self, prop, val):
		if prop.name == 'group':
			self.gdb_group = val
		elif prop.name == 'type':
			self.gdb_type = val

	def do_get_property(self, prop):
		if prop.name == 'group':
			return self.gdb_group
		elif prop.name == 'type':
			return self.gdb_type

	def do_render(self, inbuf):
		# Construct command line arguments for gracedb client
		cmdline = 'gracedb %s %s -' % (self.gdb_group, self.gdb_type)

		# Loop over all coincidences in the buffer.
		for sngls in pipeio.sngl_inspiral_groups_from_buffer(inbuf):

			# Make XML document.
			xmldoc = ligolw.Document()

			# Append the LIGO_LW tag.
			xmldoc.appendChild(ligolw.LIGO_LW())

			# Add a SnglInspiralTable.
			sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable)
			xmldoc.childNodes[0].appendChild(sngl_inspiral_table)

			# Add CoincEventTable.
			coinc_event_table = lsctables.New(lsctables.CoincTable)
			xmldoc.childNodes[0].appendChild(coinc_event_table)

			# Add CoincDefinerTable.
			coinc_definer_table = lsctables.New(lsctables.CoincDefTable)
			xmldoc.childNodes[0].appendChild(coinc_definer_table)

			# Add a CoincInspiralTable.
			coinc_inspiral_table = lsctables.New(lsctables.CoincInspiralTable)
			xmldoc.childNodes[0].appendChild(coinc_inspiral_table)

			# Add a CoincEventMapTable.
			coinc_event_map_table = lsctables.New(lsctables.CoincMapTable)
			xmldoc.childNodes[0].appendChild(coinc_event_map_table)

			mean_end_time = sum([LIGOTimeGPS(sngl.end_time, sngl.end_time_ns) for sngl in sngls]) / len(sngls)

			# Fill in rows of tables.
			coinc_definer = coinc_definer_table.RowType(coinc_def_id = coinc_definer_table.get_next_id(), search = u"inspiral", search_coinc_type = 0, description = u"sngl_inspiral<-->sngl_inspiral coincidences")
			coinc_definer_table.append(coinc_definer)
			coinc_event = coinc_event_table.RowType()
			coinc_event.coinc_event_id = coinc_event_table.get_next_id()
			coinc_event.instruments = None
			coinc_event.nevents = len(sngls)
			coinc_event.process_id = sngls[0].process_id
			coinc_event.time_slide_id = None
			coinc_event.likelihood = None
			coinc_event.coinc_def_id = coinc_definer.coinc_def_id
			coinc_event_table.append(coinc_event)
			coinc_inspiral = coinc_inspiral_table.RowType()
			coinc_inspiral.coinc_event_id = coinc_event.coinc_event_id
			coinc_inspiral.mchirp = sngls[0].mchirp
			coinc_inspiral.mass = sngls[0].mtotal
			coinc_inspiral.false_alarm_rate = None
			coinc_inspiral.combined_far = None
			coinc_inspiral.end_time = mean_end_time.seconds
			coinc_inspiral.end_time_ns = mean_end_time.nanoseconds
			coinc_inspiral.ifos = None
			coinc_inspiral.snr = numpy.sqrt(sum([sngl.snr**2 for sngl in sngls]))
			coinc_inspiral_table.append(coinc_inspiral)

			for i, sngl in enumerate(sngls):
				sngl.event_id = lsctables.SnglInspiralID(i)
				sngl_inspiral_table.append(sngl)
				coinc_event_map = coinc_event_map_table.RowType()
				coinc_event_map.coinc_event_id = coinc_event.coinc_event_id
				coinc_event_map.table_name = sngl_inspiral_table.tableName
				coinc_event_map.event_id = sngl.event_id
				coinc_event_map_table.append(coinc_event_map)

			# Write XML file to string
			strio = StringIO()
			xmldoc.write(strio)

			# Open subprocess to submit event to GraCeDB
			proc = subprocess.Popen(cmdline, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
			(stdoutdata, stderrdata) = proc.communicate(strio.getvalue())

			# Check return code
			if proc.returncode != 0:
				# FIXME: should we also return gst.FLOW_ERROR?
				self.error('gracedb exited with return code %d' % coinc_process.returncode)

			# Report gracedb ID
			self.info('gracedb ID: %s' % stdoutdata)

			# Discard XML file
			strio.close()

		# Done.
		return gst.FLOW_OK


# Register element class
gstlal_element_register(lal_gracedbsink)
