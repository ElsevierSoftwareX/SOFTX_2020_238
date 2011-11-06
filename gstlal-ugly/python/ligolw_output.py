#
# Copyright (C) 2010-2011
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


import threading
import time
import os
try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import sys

from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import ligolw_add
from glue.ligolw.utils import process as ligolw_process
from pylal.datatypes import LIGOTimeGPS
from pylal.date import XLALUTCToGPS
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer
from pylal import ligolw_burca_tailor
from pylal import ligolw_tisi
from pylal import rate
lsctables.LIGOTimeGPS = LIGOTimeGPS


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
		val = getattr(options, key)
		if val is not None and val is not False:
			opt = getattr(options, key, "")
			if isinstance(opt,list): opt = ",".join(opt)
			params[key] = opt

	return params


#
# Parameter distributions
#


class DistributionsStats(object):
	"""
	A class used to populate a CoincParamsDistribution instance using
	event parameter data.
	"""

	binnings = {
		"H1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(3., 100., 500), rate.LogarithmicPlusOverflowBins(.1, 1., 500))),
		"H2_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(3., 100., 500), rate.LogarithmicPlusOverflowBins(.1, 1., 500))),
		"L1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(3., 100., 500), rate.LogarithmicPlusOverflowBins(.1, 1., 500))),
		"V1_snr_chi": rate.NDBins((rate.LogarithmicPlusOverflowBins(3., 100., 500), rate.LogarithmicPlusOverflowBins(.1, 1., 500)))
	}

	filters = {
		"H1_snr_chi": rate.gaussian_window2d(5, 5, sigma = 20),
		"H2_snr_chi": rate.gaussian_window2d(5, 5, sigma = 20),
		"L1_snr_chi": rate.gaussian_window2d(5, 5, sigma = 20),
		"V1_snr_chi": rate.gaussian_window2d(5, 5, sigma = 20)
	}

	def __init__(self):
		self.distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)

	def add_single(self, event):
		self.distributions.add_background({
			("%s_snr_chi" % event.ifo): (event.snr, event.chisq**.5 / event.snr)
		})

	def finish(self):
		self.distributions.finish(filters = self.filters)


def get_coincparamsdistributions(xmldoc):
	# FIXME:  copied from pylal.stringutils.  make one version that can
	# be re-used
	coincparamsdistributions, process_id = ligolw_burca_tailor.coinc_params_distributions_from_xml(xmldoc, u"gstlal_inspiral_likelihood")
	seglists = lsctables.table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName).get_out_segmentlistdict(set([process_id])).coalesce()
	return coincparamsdistributions, seglists


def load_likelihood_data(filenames, verbose = False):
	# FIXME:  copied from pylal.stringutils.  make one version that can
	# be re-used
	coincparamsdistributions = None
	for n, filename in enumerate(filenames):
		if verbose:
			print >>sys.stderr, "%d/%d:" % (n + 1, len(filenames)),
		xmldoc = utils.load_filename(filename, verbose = verbose)
		if coincparamsdistributions is None:
			coincparamsdistributions, seglists = get_coincparamsdistributions(xmldoc)
		else:
			a, b = get_coincparamsdistributions(xmldoc)
			coincparamsdistributions += a
			seglists |= b
			del a, b
		xmldoc.unlink()
	return coincparamsdistributions, seglists


def write_likelihood_data(filename, coincparamsdistributions, seglists, verbose = False):
	utils.write_filename(ligolw_burca_tailor.gen_likelihood_control(coincparamsdistributions, seglists, name = u"gstlal_inspiral_likelihood"), filename, verbose = verbose, gz = (filename or "stdout").endswith(".gz"))


#
# Output document
#


class Data(object):
	def __init__(self, filename, process_params, instruments, seg, out_seg, injection_filename = None, time_slide_file = None, comment = None, tmp_path = None, verbose = False):
		self.lock = threading.Lock()
		self.filename = filename
		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, "gstlal_inspiral", process_params, comment = comment, ifos = instruments)
		self.search_summary = add_cbc_metadata(self.xmldoc, self.process, seg, out_seg)
		self.process_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessTable.tableName)
		self.process_params_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessParamsTable.tableName)
		# FIXME:  argh, ugly
		self.sngl_inspiral_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "event_id")))
		self.coinc_definer_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		self.coinc_event_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		self.coinc_event_map_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		self.time_slide_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		self.coinc_inspiral_table = self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))

		# record instruments
		self.instruments = instruments

		# if we have a time slide table file, add it.  Otherwise,
		# add an all-zero offset vector to the time_slide table
		if time_slide_file is not None:
			ligolw_add.ligolw_add(self.xmldoc, [time_slide_file], verbose = verbose)
		else:
			for row in ligolw_tisi.RowsFromOffsetDict(dict.fromkeys(instruments, 0.0), self.time_slide_table.get_next_id(), self.process):
				self.time_slide_table.append(row)

		self.sngl_inspiral_table.set_next_id(lsctables.SnglInspiralID(0))	# FIXME:  remove when lsctables.py has an ID generator attached to sngl_inspiral table

		# setup histograms
		self.distribution_stats = DistributionsStats()

		# Add injections table if necessary
		if injection_filename is not None:
			ligolw_add.ligolw_add(self.xmldoc, [injection_filename], verbose = verbose)

		if filename is not None and filename.endswith('.sqlite'):
			from glue.ligolw.utils import ligolw_sqlite
			from glue.ligolw import dbtables
			self.working_filename = dbtables.get_connection_filename(filename, tmp_path = tmp_path, replace_file = True, verbose = verbose)
			self.connection = sqlite3.connect(self.working_filename, check_same_thread=False)
			ligolw_sqlite.insert_from_xmldoc(self.connection, self.xmldoc, preserve_ids = True, verbose = verbose)

			self.xmldoc.removeChild(self.xmldoc.childNodes[-1]).unlink()
			self.xmldoc.appendChild(dbtables.get_xml(self.connection))
			self.process_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessTable.tableName)
			self.process_params_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessParamsTable.tableName)
			self.sngl_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.SnglInspiralTable.tableName)
			self.coinc_definer_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincDefTable.tableName)
			self.coinc_event_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincTable.tableName)
			self.coinc_event_map_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincMapTable.tableName)
			self.time_slide_table = lsctables.table.get_table(self.xmldoc, lsctables.TimeSlideTable.tableName)
			self.coinc_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincInspiralTable.tableName)
		else:
			self.connection = None

	def appsink_new_buffer(self, elem):
		self.lock.acquire()
		for row in sngl_inspirals_from_buffer(elem.emit("pull-buffer")):
			if LIGOTimeGPS(row.end_time, row.end_time_ns) in self.search_summary.get_out():
				row.process_id = self.process.process_id
				row.event_id = self.sngl_inspiral_table.get_next_id()
				self.sngl_inspiral_table.append(row)
				# update the parameter distribution data
				self.distribution_stats.add_single(row)
		if self.connection is not None:
			self.connection.commit()
		self.lock.release()

	def write_output_file(self, verbose = False):
		if self.connection is not None:
			from glue.ligolw import dbtables
			self.connection.cursor().execute('UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral)')
			self.connection.cursor().execute('UPDATE process SET end_time = ?', (XLALUTCToGPS(time.gmtime()).seconds,))
			self.connection.commit()
			dbtables.build_indexes(self.connection, verbose = verbose)
			dbtables.put_connection_filename(self.filename, self.working_filename, verbose = verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			self.search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_process.set_process_end_time(self.process)
			utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose)

		# write out the snr / chisq histograms
		fname = os.path.split(self.filename)
		fname = os.path.join(fname[0], '%s_snr_chi.xml.gz' % ('.'.join(fname[1].split('.')[:-1]),))
		write_likelihood_data(fname, self.distribution_stats.distributions, segments.segmentlistdict.fromkeys(self.instruments, segments.segmentlist([self.search_summary.get_out()])), verbose = verbose)


#
# Tool to split XML document tree containing sngl_inspiral coincs into a
# sequence of XML document trees each containing a single coinc
#


def split_sngl_inspiral_coinc_xmldoc(xmldoc):
	#
	# index the process, process params and search_summary tables
	#

	process_table = lsctables.table.get_table(xmldoc, lsctables.ProcessTable.tableName)
	process_params_table = lsctables.table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName)
	search_summary_table = lsctables.table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName)

	process_index = dict((row.process_id, row) for row in process_table)
	process_params_index = {}
	for row in process_params_table:
		process_params_index.setdefault(row.process_id, []).append(row)
	search_summary_index = dict((row.process_id, row) for row in search_summary_table)

	#
	# index the sngl_inspiral table
	#

	sngl_inspiral_table = lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	sngl_inspiral_index = dict((row.event_id, row) for row in sngl_inspiral_table)

	#
	# find the sngl_inspiral<-->sngl_inspiral coincs
	#

	coinc_def_table = lsctables.table.get_table(xmldoc, lsctables.CoincDefTable.tableName)
	coinc_event_table = lsctables.table.get_table(xmldoc, lsctables.CoincTable.tableName)
	coinc_event_map_table = lsctables.table.get_table(xmldoc, lsctables.CoincMapTable.tableName)
	time_slide_table = lsctables.table.get_table(xmldoc, lsctables.TimeSlideTable.tableName)

	coinc_def, = (row for row in coinc_def_table if row.search == "inspiral" and row.search_coinc_type == 0)
	coinc_event_map_index = dict((row.coinc_event_id, []) for row in coinc_event_table if row.coinc_def_id == coinc_def.coinc_def_id)
	for row in coinc_event_map_table:
		try:
			coinc = coinc_event_map_index[row.coinc_event_id]
		except KeyError:
			continue
		coinc.append(row)
	time_slide_index = dict((time_slide_id, []) for time_slide_id in set(time_slide_table.getColumnByName("time_slide_id")))
	for row in time_slide_table:
		time_slide_index[row.time_slide_id].append(row)

	for row in coinc_event_table:
		if row.coinc_def_id != coinc_def.coinc_def_id:
			continue

		newxmldoc = ligolw.Document()
		newxmldoc.appendChild(ligolw.LIGO_LW())

		new_process_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(process_table))
		new_process_params_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(process_params_table))
		new_search_summary_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(search_summary_table))
		new_sngl_inspiral_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(sngl_inspiral_table))
		new_coinc_def_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(coinc_def_table))
		new_coinc_event_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(coinc_event_table))
		new_coinc_event_map_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(coinc_event_map_table))
		new_time_slide_table = newxmldoc.childNodes[-1].appendChild(lsctables.table.new_from_template(time_slide_table))

		new_coinc_def_table.append(coinc_def)
		new_coinc_event_table.append(row)
		new_coinc_event_map_table.extend(coinc_event_map_index[row.coinc_event_id])
		new_time_slide_table.extend(time_slide_index[row.time_slide_id])
		for row in new_coinc_event_map_table:
			new_sngl_inspiral_table.append(sngl_inspiral_index[row.event_id])

		process_ids = set(row.process_id for row in new_sngl_inspiral_table) | set(row.process_id for row in new_coinc_event_table)
		for process_id in process_ids:
			new_process_table.append(process_index[process_id])
			new_process_params_table.extend(process_params_index[process_id])
			new_search_summary_table.append(search_summary_index[process_id])

		yield newxmldoc
