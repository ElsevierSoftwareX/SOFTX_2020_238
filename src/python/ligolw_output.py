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
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from pylal.datatypes import LIGOTimeGPS
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer


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

	#
	# required options
	#

	for option in ("gps_start_time", "gps_end_time", "instrument", "channel_name", "output"):
		params[option] = getattr(options, option)
	# FIXME:  what about template_bank?

	#
	# optional options
	#

	for option in ("frame_cache", "injections", "flow", "svd_tolerance", "reference_psd", "ortho_gate_fap", "snr_threshold", "write_pipeline", "write_psd", "fake_data", "online_data", "comment", "verbose"):
		if getattr(options, option) is not None:
			params[option] = getattr(options, option)

	#
	# done
	#

	return list(ligolw_process.process_params_from_dict(params))

class Data(object):
	def __init__(self, options, detectors):
		self.detectors = detectors
		self.tmp_space = options.tmp_space
		self.xmldoc = None
		self.output = options.output
		self.seg = options.seg
		self.out_seg = options.out_seg
		self.injections = options.injections
		self.comment = options.comment
		self.verbose = options.verbose
		self.sngl_inspiral_table = None
		self.seg = options.seg
		self.injection_file = options.injections
		self.output = options.output
		self.connection = None

	def prepare_output_file(self, process_params):
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.append_process(xmldoc, program = "gstlal_inspiral", comment = self.comment, ifos = set(self.detectors))
		ligolw_process.append_process_params(xmldoc, self.process, process_params)
		search_summary = add_cbc_metadata(xmldoc, self.process, self.seg, self.out_seg)
		# FIXME:  argh, ugly
		sngl_inspiral_table = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "event_id")))

		sngl_inspiral_table.set_next_id(lsctables.SnglInspiralID(0))	# FIXME:  remove when lsctables.py has an ID generator attached to sngl_inspiral table

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
			if self.injection_file is not None:
				ligolw_sqlite.insert_from_url(self.connection, self.injection_file, preserve_ids = False, verbose = self.verbose)
				#utils.load_filename(self.injection_file, gz = (injection_file or "stdin").endswith(".gz"), verbose = self.verbose).unlink()
			self.xmldoc = dbtables.get_xml(self.connection)
			self.sngl_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.SnglInspiralTable.tableName)
		else:
			from glue.ligolw.utils import ligolw_add
			self.xmldoc = xmldoc
			self.sngl_inspiral_table = sngl_inspiral_table
			if self.injection_file is not None:
				ligolw_add.ligolw_add(self.xmldoc, [self.injection_file], verbose = self.verbose)
				utils.load_filename(self.injection_file, gz = (self.injection_file or "stdin").endswith(".gz"), verbose = self.verbose)

	def write_output_file(self):
		if self.connection:
			from glue.ligolw import dbtables
			from pylal.date import XLALUTCToGPS
			import time
			self.connection.cursor().execute('UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral)')
			self.connection.cursor().execute('UPDATE process SET end_time = ?', (XLALUTCToGPS(time.gmtime()).seconds,))
			self.connection.commit()
			dbtables.build_indexes(self.connection, options.verbose)
			dbtables.put_connection_filename(self.output, self.working_filename, verbose = self.verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			search_summary = lsctables.table.get_table(self.xmldoc, lsctables.SearchSummaryTable.tableName)
			search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_process.set_process_end_time(self.process)
			utils.write_filename(self.xmldoc, self.output, gz = (self.output or "stdout").endswith(".gz"), verbose = self.verbose)

