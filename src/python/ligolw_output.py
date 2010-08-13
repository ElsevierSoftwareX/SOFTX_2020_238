#
# Copyright (C) 2010
# Kipp Cannon <kipp.cannon@ligo.org>
# Chad Hanna <chad.hanna@ligo.org>
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

try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import math
import numpy
from optparse import OptionParser
import sys
import os.path
import os

from gstlal.pipeutil import *
from gstlal.lloidparts import *

from glue import segments
from glue import segmentsUtils
from glue import lal
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from pylal.datatypes import LIGOTimeGPS
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer


#
# Utilities
#


def mchirp(m1,m2):
	return (m1+m2)**(0.6) / (m1*m2)**(0.2)


#
# add metadata to an xml document in the style of lalapps_inspiral
#


def add_cbc_metadata(xmldoc, process, seg_in, seg_out):
	#
	# add entry to search_summary table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SearchSummaryTable))
	search_summary = tbl.RowType()
	search_summary.process_id = process.process_id
	search_summary.shared_object = None # FIXME
	search_summary.lalwrapper_cvs_tag = None # FIXME
	search_summary.lal_cvs_tag = None # FIXME
	search_summary.comment = process.comment
	search_summary.set_ifos(process.get_ifos())
	search_summary.set_in(seg_in)
	search_summary.set_out(seg_out)
	search_summary.nevents = None # FIXME
	search_summary.nnodes = 1
	tbl.append(search_summary)

	#
	# add entry to filter table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.FilterTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.FilterTable))
	tbl.sync_next_id()
	row = tbl.RowType()
	row.process_id = process.process_id
	row.program = process.program
	row.start_time = int(seg_in[0])
	row.filter_name = None # FIXME
	row.filter_id = tbl.get_next_id()
	row.param_set = None # FIXME
	row.comment = process.comment
	tbl.append(row)

	#
	# add entries to search_summvars table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SearchSummVarsTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SearchSummVarsTable))
	tbl.sync_next_id()
	# FIXME

	#
	# add entries to summ_value table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SummValueTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SummValueTable))
	tbl.sync_next_id()
	# FIXME

	#
	# done
	#

	return search_summary


def make_process_params(options):
	params = {}

	for key in options.__dict__:
		if getattr(options, key) is not None:
			opt = getattr(options, key, "")
			if isinstance(opt,list): opt = ",".join(opt)
			params[key] = opt

	return list(ligolw_process.process_params_from_dict(params))

class Data(object):
	def __init__(self, detectors, options=None, **kwargs):
		self.detectors = detectors
		self.sngl_inspiral_table = None
		self.xmldoc = None
		self.connection = None

		for key in ["tmp_space", "output", "seg", "out_seg", "injections", "comment", "verbose"]:
			# keyword arguments take precedence over options
			if key in kwargs: setattr(self, key, kwargs[key])
			elif options and hasattr(options, key): setattr(self, key, getattr(options,key))
			else: raise AttributeError("must supply %s via options or key word argument" % (key,))

	def prepare_output_file(self, process_params):
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.append_process(xmldoc, program = "gstlal_inspiral", comment = self.comment, ifos = set(self.detectors))
		ligolw_process.append_process_params(xmldoc, self.process, process_params)
		search_summary = add_cbc_metadata(xmldoc, self.process, self.seg, self.out_seg)
		# FIXME:  argh, ugly
		sngl_inspiral_table = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "event_id")))

		sngl_inspiral_table.set_next_id(lsctables.SnglInspiralID(0))	# FIXME:  remove when lsctables.py has an ID generator attached to sngl_inspiral table

		if len(self.detectors) > 1:

			#
			# Coinc tables
			#

			# coinc inspiral table
			coinc_inspiral_table = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable, columns = ("coinc_event_id", "ifos", "end_time", "end_time_ns", "mass", "mchirp", "snr","false_alarm_rate","combined_far")))

			# coinc event table
			coinc_event_table = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable, columns = ("process_id","coinc_def_id","coinc_event_id","time_slide_id","instruments","nevents","likelihood")))
			coinc_event_table.set_next_id(lsctables.CoincID(0))

			# coinc event map table
			coinc_event_map = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable, columns = ("coinc_event_id","table_name", "event_id")))

			# time slide id
			time_slide_table = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable, columns = ("process_id", "time_slide_id", "instrument", "offset")))
			self.time_slide_id = lsctables.TimeSlideID(0)
			for ifo in self.detectors:
				tsrow = time_slide_table.RowType()
				tsrow.process_id = self.process.process_id
				tsrow.time_slide_id = self.time_slide_id
				tsrow.instrument = ifo
				tsrow.offset = 0.0
				time_slide_table.append(tsrow)

			# coinc def table
			coinc_definer = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable, columns = ("coinc_def_id", "search", "search_coinc_type", "description")))
			CoincDef = lsctables.CoincDef(search = u"inspiral", search_coinc_type = 0, description = u"sngl_inspiral<-->sngl_inspiral coincidences")
			CoincDef.coinc_def_id = coinc_definer.get_next_id()
			self.coinc_def_id = CoincDef.coinc_def_id
			coinc_definer.append(CoincDef)


		# Add injections table if necessary
		if self.injections is not None:
			from glue.ligolw.utils import ligolw_add
			ligolw_add.ligolw_add(xmldoc, [self.injections], verbose = self.verbose)

		if self.output.endswith('.sqlite'):
			from glue.ligolw.utils import ligolw_sqlite
			from glue.ligolw import dbtables
			self.working_filename = dbtables.get_connection_filename(self.output, tmp_path = self.tmp_space, verbose = self.verbose)
			self.connection = sqlite3.connect(self.working_filename, check_same_thread=False)
			# setup id remapping
			dbtables.idmap_create(self.connection)
			dbtables.DBTable.append = dbtables.DBTable._remapping_append
			dbtables.idmap_sync(self.connection)
			ligolw_sqlite.insert_from_xmldoc(self.connection, xmldoc, preserve_ids = False, verbose = self.verbose)
			xmldoc.unlink()
			self.xmldoc = dbtables.get_xml(self.connection)
			self.sngl_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.SnglInspiralTable.tableName)
			if len(self.detectors) > 1:
				self.coinc_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincInspiralTable.tableName)
				self.coinc_event_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincTable.tableName)
				self.coinc_event_map = lsctables.table.get_table(self.xmldoc, lsctables.CoincMapTable.tableName)
				self.time_slide_table = lsctables.table.get_table(self.xmldoc, lsctables.TimeSlideTable.tableName)
				self.coinc_definer = lsctables.table.get_table(self.xmldoc, lsctables.CoincDefTable.tableName)
		else:
			self.xmldoc = xmldoc
			self.sngl_inspiral_table = sngl_inspiral_table
			if len(self.detectors) > 1:
				self.coinc_inspiral_table = coinc_inspiral_table
				self.coinc_event_table = coinc_event_table
				self.coinc_event_map = coinc_event_map
				self.time_slide_table = time_slide_table
				self.coinc_definer = coinc_definer

	def write_output_file(self):
		if self.connection:
			from glue.ligolw import dbtables
			from pylal.date import XLALUTCToGPS
			import time
			self.connection.cursor().execute('UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral)')
			self.connection.cursor().execute('UPDATE process SET end_time = ?', (XLALUTCToGPS(time.gmtime()).seconds,))
			self.connection.commit()
			dbtables.build_indexes(self.connection, self.verbose)
			dbtables.put_connection_filename(self.output, self.working_filename, verbose = self.verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			search_summary = lsctables.table.get_table(self.xmldoc, lsctables.SearchSummaryTable.tableName)
			search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_process.set_process_end_time(self.process)
			utils.write_filename(self.xmldoc, self.output, gz = (self.output or "stdout").endswith(".gz"), verbose = self.verbose)

	def insert_group_records(self, rows):
		masses = []
		ids = []
		snrs = []
		times = []
		ifos = []
		for row in rows:
			if row.end_time != 0:
				row.process_id = self.process.process_id
				row.event_id = self.sngl_inspiral_table.get_next_id()
				self.sngl_inspiral_table.append(row)
				masses.append([row.mass1, row.mass2])
				ids.append(row.event_id)
				snrs.append(row.snr)
				ifos.append(row.ifo)
				times.append(row.end_time+row.end_time_ns/1.0e9)

		# check if we need to insert a coincidence record
		if len(ids) > 1:
			cetab = self.coinc_event_table
			row = cetab.RowType()
			row.coinc_event_id = cetab.get_next_id()
			row.nevents = len(ids)
			row.likelihood = None
			row.set_instruments(self.detectors) # FIXME wrong, we don't know what instruments were really on
			row.time_slide_id = self.time_slide_id
			row.coinc_def_id = self.coinc_def_id
			row.process_id = self.process.process_id
			cetab.append(row)

			for ceid in ids:
				cemtab = self.coinc_event_map
				cemrow = cemtab.RowType()
				cemrow.coinc_event_id = row.coinc_event_id
				cemrow.table_name = lsctables.SnglInspiralTable.tableName
				cemrow.event_id = ceid
				cemtab.append(cemrow)

			citab = self.coinc_inspiral_table
			cirow = citab.RowType()
			cirow.coinc_event_id = row.coinc_event_id
			cirow.snr = numpy.sqrt(sum([snr**2 for snr in snrs])) #FIXME
			cirow.set_ifos(ifos)
			cirow.false_alarm_rate = None #FIXME
			cirow.combined_far = None #FIXME
			# FIXME Arithmetic mean of total mass and chirpmass?
			cirow.mass = numpy.mean([m1+m2 for m1,m2 in masses])
			cirow.mchirp = numpy.mean([mchirp(m1,m2) for m1,m2 in masses])
			gps = lal.LIGOTimeGPS(min(times))
			cirow.end_time, cirow.end_time_ns = gps.seconds, gps.nanoseconds
			citab.append(cirow)






def appsink_new_buffer(elem, data):
	buf = elem.get_property("last-buffer")
	numtmps = len(data.detectors)
	rows = sngl_inspirals_from_buffer(buf)
	# if it is a multi detector pipeline then it means these are coincidence records
	for groups in [rows[i*numtmps:i*numtmps+numtmps] for i, x in enumerate(rows[::numtmps])]:
		data.insert_group_records(groups)
	if data.connection: data.connection.commit()
