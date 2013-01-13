#
# Copyright (C) 2009-2011  Kipp Cannon, Chad Hanna, Drew Keppel
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import os
from scipy import random
import StringIO
import subprocess

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import sys
import threading
import time
from collections import deque
import resource

try:
	from ligo import gracedb
except ImportError:
	print >>sys.stderr, "warning: gracedb import failed, gracedb uploads disabled"

from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import ilwd
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw import utils
from glue.ligolw.utils import ligolw_sqlite
from glue.ligolw.utils import ligolw_add
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import search_summary as ligolw_search_summary
from pylal.datatypes import LIGOTimeGPS
from pylal.datatypes import REAL8FrequencySeries
from pylal.date import XLALUTCToGPS
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer
from pylal import ligolw_tisi
from pylal import rate
from gstlal import bottle
from gstlal import reference_psd
from gstlal import streamthinca
from gstlal import svd_bank
from gstlal import far
from pylal import llwapp
from pylal import datatypes as laltypes

lsctables.LIGOTimeGPS = LIGOTimeGPS


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def channel_dict_from_channel_list(channel_list, channel_dict = {"H1" : "LSC-STRAIN", "H2" : "LSC-STRAIN", "L1" : "LSC-STRAIN", "V1" : "LSC-STRAIN"}):
	"""
	given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""

	for channel in channel_list:
		ifo = channel.split("=")[0]
		chan = "".join(channel.split("=")[1:])
		channel_dict[ifo] = chan

	return channel_dict


def pipeline_channel_list_from_channel_dict(channel_dict, opt = "channel-name"):
	"""
	produce a string of channel name arguments suitable for a pipeline.py
	program that doesn't technically allow multiple options. For example
	--channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN
	"""

	outstr = ""
	for i, ifo in enumerate(channel_dict):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[ifo])
		else:
			outstr += "--%s=%s=%s " % (opt, ifo, channel_dict[ifo])

	return outstr


def state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list, state_vector_on_off_dict = {"H1" : [0x7, 0x160], "L1" : [0x7, 0x160], "V1" : [0x67, 0x100]}):
	"""
	"""

	for line in on_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		try:
			state_vector_on_off_dict[ifo][0] = int(bits)
		except ValueError: # must be hex
			state_vector_on_off_dict[ifo][0] = int(bits, 16)
	
	for line in off_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		try:
			state_vector_on_off_dict[ifo][1] = int(bits)
		except ValueError: # must be hex
			state_vector_on_off_dict[ifo][1] = int(bits, 16)

	return state_vector_on_off_dict


def state_vector_on_off_list_from_bits_dict(bit_dict):
	"""
	"""

	onstr = ""
	offstr = ""
	for i, ifo in enumerate(bit_dict):
		if i == 0:
			onstr += "%s=%s " % (ifo, bit_dict[ifo][0])
			offstr += "%s=%s " % (ifo, bit_dict[ifo][1])
		else:
			onstr += "--state-vector-on-bits=%s=%s " % (ifo, bit_dict[ifo][0])
			offstr += "--state-vector-off-bits=%s=%s " % (ifo, bit_dict[ifo][1])

	return onstr, offstr


def parse_banks(bank_string):
	"""
	parses strings of form 
	
	H1:bank1.xml,H2:bank2.xml,L1:bank3.xml,H2:bank4.xml,... 
	
	into a dictionary of lists of bank files.
	"""
	out = {}
	if bank_string is None:
		return out
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		out.setdefault(ifo, []).append(bank)
	return out


def parse_bank_files(svd_banks, verbose, snr_threshold = None):
	"""
	given a dictionary of lists of svd template bank file names parse them
	into a dictionary of bank classes
	"""

	banks = {}

	for instrument, files in svd_banks.items():
		for n, filename in enumerate(files):
			# FIXME over ride the file name stored in the bank file with
			# this file name this bank I/O code needs to be fixed
			bank = svd_bank.read_bank(filename, contenthandler = XMLContentHandler, verbose = verbose)
			bank.template_bank_filename = filename
			bank.logname = "%sbank%d" % (instrument,n)
			banks.setdefault(instrument,[]).append(bank)
			if snr_threshold is not None:
				bank.snr_threshold = snr_threshold

	return banks


#
# add metadata to an xml document in the style of lalapps_inspiral
#


def add_cbc_metadata(xmldoc, process, seg_in):
	"""
	A convenience function to add metadata to a cbc output document
	"""
	
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
	search_summary.set_out(segments.segment(None, None))
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


#
# =============================================================================
#
#                         glue.ligolw Content Handlers
#
# =============================================================================
#


class XMLContentHandler(ligolw.LIGOLWContentHandler):
	pass

rate.array.use_in(XMLContentHandler)
rate.param.use_in(XMLContentHandler)
lsctables.use_in(XMLContentHandler)


class DBContentHandler(ligolw.LIGOLWContentHandler):
	pass
rate.array.use_in(DBContentHandler)
rate.param.use_in(DBContentHandler)
dbtables.use_in(DBContentHandler)


#
# =============================================================================
#
#                           Parameter Distributions
#
# =============================================================================
#


#
# Functions to synthesize injections
#


def snr_distribution(size, startsnr):
	"""
	This produces a power law distribution in snr of size size starting at startsnr
	"""
	return startsnr * random.power(3, size)**-1 # 3 here actually means 2 :) according to scipy docs


def noncentrality(snrs, prefactor):
	"""
	This produces a set of noncentrality parameters that scale with snr^2 according to the prefactor
	"""
	return prefactor * random.rand(len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals
	#return prefactor * random.power(1, len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals


def chisq_distribution(df, non_centralities, size):
	"""
	This produces a set of noncentral chisq values of size size, with degrees of freedom given by df
	"""
	out = numpy.empty((len(non_centralities) * size,))
	for i, nc in enumerate(non_centralities):
		out[i*size:(i+1)*size] = random.noncentral_chisquare(df, nc, size)
	return out


#
# =============================================================================
#
#                               Output Document
#
# =============================================================================
#


def gen_likelihood_control_doc(far, instruments, name = u"gstlal_inspiral_likelihood", comment = u""):
	xmldoc = ligolw.Document()
	node = xmldoc.appendChild(ligolw.LIGO_LW())

	node.appendChild(lsctables.New(lsctables.ProcessTable))
	node.appendChild(lsctables.New(lsctables.ProcessParamsTable))
	node.appendChild(lsctables.New(lsctables.SearchSummaryTable))
	process = ligolw_process.append_process(xmldoc, comment = comment)
	ligolw_search_summary.append_search_summary(xmldoc, process, ifos = instruments, inseg = far.livetime_seg, outseg = far.livetime_seg)

	node.appendChild(far.to_xml(process, name))

	llwapp.set_process_end_time(process)
	return xmldoc


class CoincsDocument(object):
	def __init__(self, filename, process_params, comment, instruments, seg, injection_filename = None, time_slide_file = None, tmp_path = None, replace_file = None, verbose = False):
		#
		# how to make another like us
		#

		self.get_another = lambda: CoincsDocument(filename = filename, process_params = process_params, comment = comment, instruments = instruments, seg = seg, injection_filename = injection_filename, time_slide_file = time_slide_file, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

		#
		# filename
		#

		self.filename = filename

		#
		# build the XML document
		#

		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, "gstlal_inspiral", process_params, comment = comment, ifos = instruments)
		self.search_summary = add_cbc_metadata(self.xmldoc, self.process, seg)
		# FIXME:  argh, ugly
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "event_id")))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))

		#
		# optionally insert injection list document
		#

		if injection_filename is not None:
			ligolw_add.ligolw_add(self.xmldoc, [injection_filename], contenthandler = XMLContentHandler, verbose = verbose)

		#
		# optionally insert a time slide table document.  if we
		# don't have one, add an all-zero offset vector.  remove
		# duplicate offset vectors when done
		#

		time_slide_table = lsctables.table.get_table(self.xmldoc, lsctables.TimeSlideTable.tableName)
		if time_slide_file is not None:
			ligolw_add.ligolw_add(self.xmldoc, [time_slide_file], contenthandler = XMLContentHandler, verbose = verbose)
		else:
			for row in ligolw_tisi.RowsFromOffsetDict(dict.fromkeys(instruments, 0.0), time_slide_table.get_next_id(), self.process):
				time_slide_table.append(row)
		time_slide_mapping = ligolw_tisi.time_slides_vacuum(time_slide_table.as_dict())
		iterutils.inplace_filter(lambda row: row.time_slide_id not in time_slide_mapping, time_slide_table)
		for tbl in self.xmldoc.getElementsByTagName(ligolw.Table.tagName):
			tbl.applyKeyMapping(time_slide_mapping)

		#
		# if the output is an sqlite database, build the sqlite
		# database and convert the in-ram XML document to an
		# interface to the database file
		#

		if filename is not None and filename.endswith('.sqlite'):
			# FIXME:  remove the ID remap stuff when we can
			# rely on having glue 1.44
			self.working_filename = dbtables.get_connection_filename(filename, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)
			self.connection = sqlite3.connect(self.working_filename, check_same_thread = False)
			dbtables.idmap_create(self.connection)
			dbtables.idmap_sync(self.connection)
			__orig_append, dbtables.DBTable.append = dbtables.DBTable.append, dbtables.DBTable._remapping_append
			ligolw_sqlite.insert_from_xmldoc(self.connection, self.xmldoc, preserve_ids = False, verbose = verbose)
			dbtables.DBTable.append = __orig_append
			del __orig_append
			dbtables.idmap_reset(self.connection)
			dbtables.idmap_sync(self.connection)

			#
			# convert self.xmldoc into wrapper interface to
			# database
			#

			self.xmldoc.removeChild(self.xmldoc.childNodes[-1]).unlink()
			self.xmldoc.appendChild(dbtables.get_xml(self.connection))

			# recover the process_id following the ID remapping
			# that might have happened when the document was
			# inserted.  hopefully this query is unique enough
			# to find exactly the one correct entry in the
			# database

			(self.process.process_id,), = (self.search_summary.process_id,), = self.connection.cursor().execute("SELECT process_id FROM process WHERE program == ? AND node == ? AND username == ? AND unix_procid == ? AND start_time == ?", (self.process.program, self.process.node, self.process.username, self.process.unix_procid, self.process.start_time)).fetchall()
			self.process.process_id = self.search_summary.process_id = ilwd.ilwdchar(self.process.process_id)
		else:
			self.connection = self.working_filename = None

		#
		# retrieve references to the table objects, now that we
		# know if they are database-backed or XML objects
		#

		self.sngl_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.SnglInspiralTable.tableName)


	def commit(self):
		# update output document
		if self.connection is not None:
			self.connection.commit()


	@property
	def process_id(self):
		return self.process.process_id


	@property
	def search_summary_outseg(self):
		return self.search_summary.get_out()


	def add_to_search_summary_outseg(self, seg):
		out_segs = segments.segmentlist([self.search_summary_outseg])
		if out_segs == [segments.segment(None, None)]:
			# out segment not yet initialized
			del out_segs[:]
		out_segs |= segments.segmentlist([seg])
		self.search_summary.set_out(out_segs.extent())


	def get_next_sngl_id(self):
		return self.sngl_inspiral_table.get_next_id()


	def T050017_filename(self, description, extension):
		start, end = self.search_summary_outseg
		return "%s-%s-%d-%d.%s" % ("".join(sorted(self.process.get_ifos())), description, int(start), int(end - start), extension)


	def write_output_file(self, verbose = False):
		ligolw_process.set_process_end_time(self.process)

		# FIXME:  should signal trapping be disabled in this code
		# path?  I think not
		if self.connection is not None:
			seg = self.search_summary_outseg
			# record the final state of the search_summary and
			# process rows in the database
			cursor = self.connection.cursor()
			if seg != segments.segment(None, None):
				cursor.execute("UPDATE search_summary SET out_start_time = ?, out_start_time_ns = ?, out_end_time = ?, out_end_time_ns = ? WHERE process_id == ?", (seg[0].seconds, seg[0].nanoseconds, seg[1].seconds, seg[1].nanoseconds, self.search_summary.process_id))
			cursor.execute("UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral) WHERE process_id == ?", (self.search_summary.process_id,))
			cursor.execute("UPDATE process SET end_time = ? WHERE process_id == ?", (self.process.end_time, self.process.process_id))
			cursor.close()
			self.connection.commit()
			dbtables.build_indexes(self.connection, verbose = verbose)
			self.connection.close()
			dbtables.put_connection_filename(self.filename, self.working_filename, verbose = verbose)
			self.connection = None
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			self.search_summary.nevents = len(self.sngl_inspiral_table)
			utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


class Data(object):
	def __init__(self, filename, process_params, pipeline, instruments, seg, coincidence_threshold, FAR, marginalized_likelihood_file = None, injection_filename = None, time_slide_file = None, comment = None, tmp_path = None, assign_likelihoods = False, likelihood_snapshot_interval = None, thinca_interval = 50.0, gracedb_far_threshold = None, likelihood_file = None, gracedb_group = "Test", gracedb_type = "LowMass", replace_file = True, verbose = False):
		#
		# initialize
		#
		
		# setup bottle routes
		bottle.route("/latency_histogram.txt")(self.web_get_latency_histogram)
		bottle.route("/latency_history.txt")(self.web_get_latency_history)
		bottle.route("/snr_history.txt")(self.web_get_snr_history)
		bottle.route("/ram_history.txt")(self.web_get_ram_history)
		bottle.route("/likelihood.xml")(self.web_get_likelihood_file)
		bottle.route("/gracedb_far_threshold.txt", method = "GET")(self.web_get_gracedb_far_threshold)
		bottle.route("/gracedb_far_threshold.txt", method = "POST")(self.web_set_gracedb_far_threshold)

		self.lock = threading.Lock()
		self.pipeline = pipeline
		self.instruments = instruments
		self.verbose = verbose
		# True to enable likelihood assignment
		self.assign_likelihoods = assign_likelihoods
		self.marginalized_likelihood_file = marginalized_likelihood_file
		# Set to None to disable period snapshots, otherwise set to seconds
		self.likelihood_snapshot_interval = likelihood_snapshot_interval
		# Setup custom checkpoint message
		appmsgstruct = gst.Structure("CHECKPOINT")
		self.checkpointmsg = gst.message_new_application(pipeline, appmsgstruct)
		# Set to 1.0 to disable background data decay
		# FIXME:  should this live in the DistributionsStats object?
		self.likelihood_snapshot_timestamp = None
		# gracedb far threshold
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_type = gracedb_type

		#
		# initialize document to hold coincs and segments
		#

		self.coincs_document = CoincsDocument(filename, process_params, comment, instruments, seg, injection_filename = injection_filename, time_slide_file = time_slide_file, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

		#
		# setup far/fap book-keeping
		#

		self.far = FAR
		self.ranking_data = None
		if self.assign_likelihoods:
			self.far.smooth_distribution_stats(verbose = verbose)
		self.likelihood_file = likelihood_file

		#
		# attach a StreamThinca instance to ourselves
		#

		self.stream_thinca = streamthinca.StreamThinca(
			coincidence_threshold = coincidence_threshold,
			thinca_interval = thinca_interval,	# seconds
			trials_table = self.far.trials_table
		)

		#
		# Fun output stuff
		#
		
		self.latency_histogram = rate.BinnedArray(rate.NDBins((rate.LinearPlusOverflowBins(5, 205, 22),)))
		self.latency_history = deque(maxlen = 1000)
		self.snr_history = deque(maxlen = 1000)
		self.ram_history = deque(maxlen = 1000)

	def appsink_new_buffer(self, elem):
		self.lock.acquire()
		try:
			# retrieve triggers from appsink element
			buf = elem.emit("pull-buffer")
			events = sngl_inspirals_from_buffer(buf)

			# update search_summary out segment.  note that
			# both the trigger document and the FAR object get
			# their own copies of the segment.  the segment in
			# self.search_summary is what gets recorded in the
			# trigger document, the segment in the FAR object
			# gets used for online FAR/FAP assignment and is
			# what gets recorded in the likelihood data
			# document.
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			buf_end_time = buf_timestamp + LIGOTimeGPS(0, buf.duration)
			self.coincs_document.add_to_search_summary_outseg(segments.segment(buf_timestamp, buf_end_time))
			if self.far.livetime_seg == segments.segment(None, None):
				self.far.livetime_seg = self.coincs_document.search_summary_outseg
			else:
				self.far.livetime_seg = segments.segmentlist([self.coincs_document.search_summary_outseg, self.far.livetime_seg]).extent()

			# set metadata on triggers.  because this uses the
			# ID generator attached to the database-backed
			# sngl_inspiral table, and that generator has been
			# synced to the database' contents, the IDs
			# assigned here will not collide with any already
			# in the database
			for event in events:
				event.process_id = self.coincs_document.process_id
				event.event_id = self.coincs_document.get_next_sngl_id()

			# update likelihood snapshot if needed
			if (self.likelihood_snapshot_timestamp is None or (self.likelihood_snapshot_interval is not None and buf_timestamp - self.likelihood_snapshot_timestamp >= self.likelihood_snapshot_interval)):
				self.likelihood_snapshot_timestamp = buf_timestamp
				# Post a checkpoint message
				self.pipeline.get_bus().post(self.checkpointmsg)
				if self.assign_likelihoods:
					assert self.marginalized_likelihood_file is not None
					# smooth the distribution_stats
					self.far.smooth_distribution_stats(verbose = self.verbose)
					# update stream thinca's likelihood data
					self.stream_thinca.set_likelihood_data(self.far.distribution_stats.smoothed_distributions, self.far.distribution_stats.likelihood_params_func)

					# Read in the the background likelihood distributions that should have been updated asynchronously
					self.ranking_data, procid = far.RankingData.from_xml(utils.load_filename(self.marginalized_likelihood_file, verbose = self.verbose, contenthandler = XMLContentHandler))
					self.ranking_data.compute_joint_cdfs()

					# set up the scale factor for the trials table to normalize the rate
					for ifos in self.ranking_data.scale:
						try:
							self.ranking_data.scale[ifos] = (self.ranking_data.trials_table[ifos].count_below_thresh or 1) / self.ranking_data.trials_table[ifos].thresh / float(abs(self.ranking_data.livetime_seg)) * (self.ranking_data.trials_table.num_nonzero_count() or 1)
						except TypeError:
							self.ranking_data.scale[ifos] = 1
							print >> sys.stderr, "could not set scale factor, probably because we do not have live time info yet.  Seg is: ", self.ranking_data.livetime_seg
							
					# write the new distribution stats to disk
					utils.write_filename(gen_likelihood_control_doc(self.far, self.instruments), self.likelihood_file, gz = (self.likelihood_file or "stdout").endswith(".gz"), verbose = False, trap_signals = None)
				else:
					self.ranking_data = None

			# run stream thinca
			noncoinc_sngls = self.stream_thinca.add_events(self.coincs_document.xmldoc, self.coincs_document.process_id, events, buf_timestamp, FAP = self.ranking_data)

			# update the parameter distribution data.  only
			# update from sngls that weren't used in coincs
			for event in noncoinc_sngls:
				self.far.distribution_stats.add_single(event)

			# update output document
			self.coincs_document.commit()

			# do GraceDB alerts
			if self.gracedb_far_threshold is not None:
				self.__do_gracedb_alerts()
				self.__update_eye_candy()
		finally:
			self.lock.release()

	def __write_likelihood_file(self):
		# write the new distribution stats to disk
		output = StringIO.StringIO()
		utils.write_fileobj(gen_likelihood_control_doc(self.far, self.instruments), output, trap_signals = None)
		outstr = output.getvalue()
		output.close()
		return outstr

	def web_get_likelihood_file(self):
		self.lock.acquire()
		try:
			outstr = self.__write_likelihood_file()
		finally:
			self.lock.release()
		return outstr

	def __flush(self):
		# run StreamThinca's .flush().  returns the last remaining
		# non-coincident sngls.  add them to the distribution
		if self.assign_likelihoods:
			FAP = self.ranking_data
		else:
			FAP = None
		for event in self.stream_thinca.flush(self.coincs_document.xmldoc, self.coincs_document.process_id, FAP = FAP):
			self.far.distribution_stats.add_single(event)
		self.coincs_document.commit()

		# do GraceDB alerts
		if self.gracedb_far_threshold is not None:
			self.__do_gracedb_alerts()

	def flush(self):
		self.lock.acquire()
		try:
			self.__flush()
		finally:
			self.lock.release()

	def __do_gracedb_alerts(self):
		try:
			gracedb
		except NameError:
			# gracedb import failed, disable event uploads
			return
		if self.stream_thinca.last_coincs:
			gracedb_client = gracedb.Client()
			gracedb_ids = []
			psdmessage = None
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index
			# FIXME:  this should maybe not be retrieved this
			# way.  and the .column_index() method is probably
			# useless
			likelihood_dict = self.stream_thinca.last_coincs.column_index(lsctables.CoincTable.tableName, "likelihood")

			# FIXME:  this is hacked to only send at most the
			# one best coinc in this set.  May not make sense
			# depending on what is in last_coincs.  FIX
			# PROPERLY.  This is probably mostly okay because
			# we should be doing coincidences every 10s which
			# is a reasonable time to cluster over.  the slice
			# can be edited (or removed) to change this.
			for likelihood, coinc_event_id in sorted((likelihood, coinc_event_id) for (coinc_event_id, likelihood) in likelihood_dict.items())[-1:]:
				#
				# quit if the false alarm rate is not low
				# enough, or is nan
				#

				if coinc_inspiral_index[coinc_event_id].combined_far > self.gracedb_far_threshold or numpy.isnan(coinc_inspiral_index[coinc_event_id].combined_far):
					break

				#
				# retrieve PSDs
				#

				if psdmessage is None:
					if self.verbose:
						print >>sys.stderr, "retrieving PSDs from whiteners and generating psd.xml.gz ..."
					psddict = {}
					for instrument in self.instruments:
						elem = self.pipeline.get_by_name("lal_whiten_%s" % instrument)
						psddict[instrument] = REAL8FrequencySeries(
							name = "PSD",
							epoch = LIGOTimeGPS(0, 0),	# FIXME
							f0 = 0.0,
							deltaF = elem.get_property("delta-f"),
							sampleUnits = laltypes.LALUnit("s strain^2"),	# FIXME:  don't hard-code this
							data = numpy.array(elem.get_property("mean-psd"))
						)
					psdmessage = StringIO.StringIO()
					reference_psd.write_psd_fileobj(psdmessage, psddict, gz = True, trap_signals = None)

				#
				# fake a filename for end-user convenience
				#

				observatories = "".join(sorted(set(instrument[0] for instrument in self.instruments)))
				instruments = "".join(sorted(self.instruments))
				description = "%s_%s_%s_%s" % (instruments, ("%.4g" % coinc_inspiral_index[coinc_event_id].mass).replace(".", "_").replace("-", "_"), self.gracedb_group, self.gracedb_type)
				end_time = int(coinc_inspiral_index[coinc_event_id].get_end())
				filename = "%s-%s-%d-%d.xml" % (observatories, description, end_time, 0)

				#
				# construct message and send to gracedb.
				# we go through the intermediate step of
				# first writing the document into a string
				# buffer incase there is some safety in
				# doing so in the event of a malformed
				# document;  instead of writing directly
				# into gracedb's input pipe and crashing
				# part way through.
				#

				if self.verbose:
					print >>sys.stderr, "sending %s to gracedb ..." % filename
				message = StringIO.StringIO()
				utils.write_fileobj(self.stream_thinca.last_coincs[coinc_event_id], message, gz = False, trap_signals = None)
				# FIXME: make this optional from command line?
				if True:
					resp = gracedb_client.create(self.gracedb_group, self.gracedb_type, filename, message.getvalue())
					if "error" in resp:
						print >>sys.stderr, "gracedb upload of %s failed: %s" % (filename, resp["error"])
					else:
						if self.verbose:
							if "warning" in resp:
								print >>sys.stderr, "gracedb issued warning: %s" % resp["warning"]
							print >>sys.stderr, "event assigned grace ID %s" % resp["graceid"]
						gracedb_ids.append(resp["graceid"])
				else:
					proc = subprocess.Popen(("/bin/cp", "/dev/stdin", filename), stdin = subprocess.PIPE)
					proc.stdin.write(message.getvalue())
					proc.stdin.flush()
					proc.stdin.close()
				message.close()

			#
			# do PSD file uploads
			#

			if psdmessage is not None:
				filename = "psd.xml.gz"
				for gracedb_id in gracedb_ids:
					resp = gracedb_client.upload(gracedb_id, filename, psdmessage.getvalue(), comment = "strain spectral densities")
					if "error" in resp:
						print >>sys.stderr, "gracedb upload of %s for ID %s failed: %s" % (filename, gracedb_id, resp["error"])

	def do_gracedb_alerts(self):
		self.lock.acquire()
		try:
			self.__do_gracedb_alerts()
		finally:
			self.lock.release()

	def __update_eye_candy(self):
		if self.stream_thinca.last_coincs:
			self.ram_history.append((time.time(), (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1048576.)) # GB
			latency_val = None
			snr_val = (0,0)
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index
			for coinc_event_id, latency in self.stream_thinca.last_coincs.column_index(lsctables.CoincInspiralTable.tableName, "minimum_duration").items():
				self.latency_histogram[latency,] += 1
				if latency_val is None:
					t = float(coinc_inspiral_index[coinc_event_id].get_end())
					latency_val = (t, latency)
				snr = coinc_inspiral_index[coinc_event_id].snr
				if snr >= snr_val[1]:
					t = float(coinc_inspiral_index[coinc_event_id].get_end())
					snr_val = (t, snr)
			if latency_val is not None:
				self.latency_history.append(latency_val)
			if snr_val != (0,0):
				self.snr_history.append(snr_val)

	def update_eye_candy(self):
		self.lock.acquire()
		try:
			self.__update_eye_candy()
		finally:
			self.lock.release()


	def web_get_latency_histogram(self):
		self.lock.acquire()
		try:
			for latency, number in zip(self.latency_histogram.centres()[0][1:-1], self.latency_histogram.array[1:-1]):
				yield "%e %e\n" % (latency, number)
		finally:
			self.lock.release()


	def web_get_latency_history(self):
		self.lock.acquire()
		try:
			# first one in the list is sacrificed for a time stamp
			for time, latency in self.latency_history:
				yield "%f %e\n" % (time, latency)
		finally:
			self.lock.release()


	def web_get_snr_history(self):
		self.lock.acquire()
		try:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.snr_history:
				yield "%f %e\n" % (time, snr)
		finally:
			self.lock.release()


	def web_get_ram_history(self):
		self.lock.acquire()
		try:
			# first one in the list is sacrificed for a time stamp
			for time, ram in self.ram_history:
				yield "%f %e\n" % (time, ram)
		finally:
			self.lock.release()


	def web_get_gracedb_far_threshold(self):
		self.lock.acquire()
		try:
			if self.gracedb_far_threshold is not None:
				yield "rate=%.17g\n" % self.gracedb_far_threshold
			else:
				yield "rate=\n"
		finally:
			self.lock.release()


	def web_set_gracedb_far_threshold(self):
		self.lock.acquire()
		try:
			rate = bottle.request.forms["rate"]
			if rate:
				self.gracedb_far_threshold = float(rate)
				yield "OK: rate=%.17g\n" % self.gracedb_far_threshold
			else:
				self.gracedb_far_threshold = None
				yield "OK: rate=\n"
		except:
			yield "error\n"
		finally:
			self.lock.release()


	def __write_output_file(self, filename = None, likelihood_file = None, verbose = False):
		self.__flush()
		if filename is not None:
			self.coincs_document.filename = filename
		self.coincs_document.write_output_file(verbose = verbose)

		# write out the snr / chisq histograms
		if likelihood_file is None:
			fname = os.path.split(self.coincs_document.filename)
			ifo, desc, start, dur = ".".join(fname[1].split('.')[:-1]).split('-')
			fname = os.path.join(fname[0], '%s-%s_SNR_CHI-%s-%s.xml.gz' % (ifo, desc, start, dur))
		else:
			fname = likelihood_file
		utils.write_filename(gen_likelihood_control_doc(self.far, self.instruments), fname, gz = (fname or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


	def write_output_file(self, filename = None, likelihood_file = None, verbose = False):
		self.lock.acquire()
		try:
			self.__write_output_file(filename = filename, likelihood_file = likelihood_file, verbose = verbose)
			# can't be used anymore
			del self.coincs_document
		finally:
			self.lock.release()


	def snapshot_output_file(self, description, extension, verbose = False):
		self.lock.acquire()
		try:
			self.__write_output_file(filename = self.coincs_document.T050017_filename(description, extension), verbose = verbose)
			self.coincs_document = self.coincs_document.get_another()
		finally:
			self.lock.release()
