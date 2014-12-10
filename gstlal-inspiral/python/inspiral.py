# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
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


from collections import deque
import copy
import math
import numpy
import os
import resource
from scipy import random
try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import StringIO
import subprocess
import sys
import threading
import time
import httplib
import tempfile

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

try:
	from ligo import gracedb
except ImportError:
	print >>sys.stderr, "warning: gracedb import failed, program will crash if gracedb uploads are attempted"

from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import dbtables
from glue.ligolw import ilwd
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import ligolw_sqlite
from glue.ligolw.utils import ligolw_add
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import time_slide as ligolw_time_slide
from pylal import datatypes as laltypes
from pylal import rate
from pylal.datatypes import LIGOTimeGPS
from pylal.datatypes import REAL8FrequencySeries
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer

from gstlal import bottle
from gstlal import reference_psd
from gstlal import streamthinca
from gstlal import svd_bank
from gstlal import cbc_template_iir
from gstlal import far

lsctables.LIGOTimeGPS = LIGOTimeGPS


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def message_new_checkpoint(src, timestamp = None):
	s = gst.Structure("CHECKPOINT")
	s.set_value("timestamp", timestamp)
	return gst.message_new_application(src, s)


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


def parse_svdbank_string(bank_string):
	"""
	parses strings of form 
	
	H1:bank1.xml,H2:bank2.xml,L1:bank3.xml
	
	into a dictionary of lists of bank files.
	"""
	out = {}
	if bank_string is None:
		return out
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		if ifo in out:
			raise ValueError("Only one svd bank per instrument should be given")
		out[ifo] = bank
	return out


def parse_iirbank_string(bank_string):
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

	for instrument, filename in svd_banks.items():
		for n, bank in enumerate(svd_bank.read_banks(filename, contenthandler = LIGOLWContentHandler, verbose = verbose)):
			# Write out sngl inspiral table to temp file for trigger generator
			# FIXME teach the trigger generator to get this information a better way
			bank.template_bank_filename = tempfile.NamedTemporaryFile(suffix = ".gz", delete = False).name
			xmldoc = ligolw.Document()
			# FIXME if this table reference is from a DB this is a problem (but it almost certainly isn't)
			xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(bank.sngl_inspiral_table.copy()).extend(bank.sngl_inspiral_table)
			ligolw_utils.write_filename(xmldoc, bank.template_bank_filename, gz = True, verbose = verbose)
			xmldoc.unlink()	# help garbage collector
			bank.logname = "%sbank%d" % (instrument, n)
			banks.setdefault(instrument, []).append(bank)
			if snr_threshold is not None:
				bank.snr_threshold = snr_threshold

	# FIXME remove when this is no longer an issue
	if not banks:
		raise ValueError("Could not parse bank files into valid bank dictionary.\n\t- Perhaps you are using out-of-date svd bank files?  Please ensure that they were generated with the same code version as the parsing code")
	return banks

def parse_iirbank_files(iir_banks, verbose, snr_threshold = 4.0):
	"""
	given a dictionary of lists of iir template bank file names parse them
	into a dictionary of bank classes
	"""

	banks = {}

	for instrument, files in iir_banks.items():
		for n, filename in enumerate(files):
			# FIXME over ride the file name stored in the bank file with
			# this file name this bank I/O code needs to be fixed
			bank = cbc_template_iir.load_iirbank(filename, snr_threshold, contenthandler = LIGOLWContentHandler, verbose = verbose)
			bank.template_bank_filename = filename
			bank.logname = "%sbank%d" % (instrument,n)
			banks.setdefault(instrument,[]).append(bank)

	return banks


#
# =============================================================================
#
#                         glue.ligolw Content Handlers
#
# =============================================================================
#


class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass
ligolw_array.use_in(LIGOLWContentHandler)
ligolw_param.use_in(LIGOLWContentHandler)
lsctables.use_in(LIGOLWContentHandler)


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


class CoincsDocument(object):
	sngl_inspiral_columns = ("process_id", "ifo", "end_time", "end_time_ns", "eff_distance", "coa_phase", "mass1", "mass2", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "sigmasq", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id")

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
		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, u"gstlal_inspiral", process_params, comment = comment, ifos = instruments)
		self.search_summary = ligolw_search_summary.append_search_summary(self.xmldoc, self.process,
			lalwrapper_cvs_tag = None,	# FIXME
			lal_cvs_tag = None,	# FIXME
			inseg = seg
		)
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = self.sngl_inspiral_columns))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentDefTable, columns = ligolw_segments.LigolwSegmentList.segment_def_columns))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentSumTable, columns = ligolw_segments.LigolwSegmentList.segment_sum_columns))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentTable, columns = ligolw_segments.LigolwSegmentList.segment_columns))

		#
		# optionally insert injection list document
		#

		if injection_filename is not None:
			ligolw_add.ligolw_add(self.xmldoc, [injection_filename], contenthandler = LIGOLWContentHandler, verbose = verbose)

		#
		# optionally insert a time slide table document.  if we
		# don't have one, add an all-zero offset vector.  remove
		# duplicate offset vectors when done
		#

		time_slide_table = lsctables.TimeSlideTable.get_table(self.xmldoc)
		if time_slide_file is not None:
			ligolw_add.ligolw_add(self.xmldoc, [time_slide_file], contenthandler = LIGOLWContentHandler, verbose = verbose)
		else:
			time_slide_table.append_offsetvector(dict.fromkeys(instruments, 0.0), self.process)
		time_slide_mapping = ligolw_time_slide.time_slides_vacuum(time_slide_table.as_dict())
		iterutils.inplace_filter(lambda row: row.time_slide_id not in time_slide_mapping, time_slide_table)
		for tbl in self.xmldoc.getElementsByTagName(ligolw.Table.tagName):
			tbl.applyKeyMapping(time_slide_mapping)

		#
		# if the output is an sqlite database, build the sqlite
		# database and convert the in-ram XML document to an
		# interface to the database file
		#

		if filename is not None and filename.endswith('.sqlite'):
			self.working_filename = dbtables.get_connection_filename(filename, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)
			self.connection = sqlite3.connect(self.working_filename, check_same_thread = False)
			ligolw_sqlite.insert_from_xmldoc(self.connection, self.xmldoc, preserve_ids = False, verbose = verbose)

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

		self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(self.xmldoc)
		self.llwsegments = ligolw_segments.LigolwSegments(self.xmldoc, self.process)


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
		start, end = int(math.floor(start)), int(math.ceil(end))
		return "%s-%s-%d-%d.%s" % ("".join(sorted(self.process.get_ifos())), description, start, end - start, extension)


	def write_output_file(self, verbose = False):
		self.llwsegments.finalize()
		ligolw_process.set_process_end_time(self.process)

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
			self.connection = None
			dbtables.put_connection_filename(self.filename, self.working_filename, verbose = verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			self.search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


class Data(object):
	def __init__(self, filename, process_params, pipeline, instruments, seg, coincidence_threshold, coinc_params_distributions, ranking_data, marginalized_likelihood_file = None, likelihood_file = None, injection_filename = None, time_slide_file = None, comment = None, tmp_path = None, likelihood_snapshot_interval = None, thinca_interval = 50.0, sngls_snr_threshold = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal", replace_file = True, verbose = False):
		#
		# initialize
		#

		self.lock = threading.Lock()
		self.pipeline = pipeline
		self.verbose = verbose
		# None to disable likelihood ratio assignment, otherwise a filename
		self.marginalized_likelihood_file = marginalized_likelihood_file
		self.likelihood_file = likelihood_file
		# None to disable periodic snapshots, otherwise seconds
		# set to 1.0 to disable background data decay
		self.likelihood_snapshot_interval = likelihood_snapshot_interval
		self.likelihood_snapshot_timestamp = None
		# gracedb far threshold
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline

		#
		# setup bottle routes
		#

		bottle.route("/latency_histogram.txt")(self.web_get_latency_histogram)
		bottle.route("/latency_history.txt")(self.web_get_latency_history)
		bottle.route("/snr_history.txt")(self.web_get_snr_history)
		bottle.route("/ram_history.txt")(self.web_get_ram_history)
		bottle.route("/likelihood.xml")(self.web_get_likelihood_file)
		bottle.route("/gracedb_far_threshold.txt", method = "GET")(self.web_get_gracedb_far_threshold)
		bottle.route("/gracedb_far_threshold.txt", method = "POST")(self.web_set_gracedb_far_threshold)
		bottle.route("/sngls_snr_threshold.txt", method = "GET")(self.web_get_sngls_snr_threshold)
		bottle.route("/sngls_snr_threshold.txt", method = "POST")(self.web_set_sngls_snr_threshold)

		#
		# initialize document to hold coincs and segments
		#

		self.coincs_document = CoincsDocument(filename, process_params, comment, instruments, seg, injection_filename = injection_filename, time_slide_file = time_slide_file, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

		#
		# attach a StreamThinca instance to ourselves
		#

		self.stream_thinca = streamthinca.StreamThinca(
			coincidence_threshold = coincidence_threshold,
			thinca_interval = thinca_interval,	# seconds
			sngls_snr_threshold = sngls_snr_threshold
		)

		#
		# setup likelihood ratio book-keeping.  seglistsdicts to be
		# populated the pipeline handler
		#

		self.coinc_params_distributions = coinc_params_distributions
		self.ranking_data = ranking_data
		self.seglistdicts = None
		self.fapfar = None

		#
		# Fun output stuff
		#
		
		self.latency_histogram = rate.BinnedArray(rate.NDBins((rate.LinearPlusOverflowBins(5, 205, 22),)))
		self.latency_history = deque(maxlen = 1000)
		self.snr_history = deque(maxlen = 1000)
		self.ram_history = deque(maxlen = 1000)

	def appsink_new_buffer(self, elem):
		with self.lock:
			# retrieve triggers from appsink element
			buf = elem.emit("pull-buffer")
			events = sngl_inspirals_from_buffer(buf)
			# FIXME:  ugly way to get the instrument
			instrument = elem.get_name().split("_")[0]

			# update search_summary out segment and our
			# livetime
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			buf_seg = segments.segment(buf_timestamp, buf_timestamp + LIGOTimeGPS(0, buf.duration))
			self.coincs_document.add_to_search_summary_outseg(buf_seg)
			self.seglistdicts["triggersegments"][instrument] |= segments.segmentlist((buf_seg,))

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
			if self.likelihood_snapshot_interval is not None and (self.likelihood_snapshot_timestamp is None or buf_timestamp - self.likelihood_snapshot_timestamp >= self.likelihood_snapshot_interval):
				self.likelihood_snapshot_timestamp = buf_timestamp

				# smooth the distributions.  re-populates
				# PDF arrays from raw counts
				self.coinc_params_distributions.finish(verbose = self.verbose)

				# post a checkpoint message.  FIXME:  make
				# sure this triggers
				# self.snapshot_output_file() to be
				# invoked.  lloidparts takes care of that
				# for now, but spreading the program logic
				# around like that isn't a good idea, this
				# code should be responsible for it
				# somehow, no?
				self.pipeline.get_bus().post(message_new_checkpoint(self.pipeline, timestamp = buf_timestamp.ns()))

				if self.marginalized_likelihood_file is not None:
					# FIXME:  must set horizon
					# distances in coinc params object

					# enable streamthinca's likelihood
					# ratio assignment using our own,
					# local, parameter distribution
					# data
					self.stream_thinca.coinc_params_distributions = self.coinc_params_distributions

					# read the marginalized likelihood
					# ratio distributions that have
					# been updated asynchronously and
					# initialize a FAP/FAR assignment
					# machine from it.  NOTE:  to keep
					# overhead low we do not .finish()
					# the ranking data here;  the
					# external process generating this
					# input file must do that for us.
					coinc_params_distributions, ranking_data, seglists = far.parse_likelihood_control_doc(ligolw_utils.load_filename(self.marginalized_likelihood_file, verbose = self.verbose, contenthandler = far.ThincaCoincParamsDistributions.LIGOLWContentHandler))
					if coinc_params_distributions is None:
						raise ValueError("\"%s\" does not contain coinc parameter PDFs" % self.marginalized_likelihood_file)
					if ranking_data is None:
						raise ValueError("\"%s\" does not contain ranking statistic PDFs" % self.marginalized_likelihood_file)
					# we're using the class attribute
					# elsewhere so make sure these two
					# match
					assert ranking_data.ln_likelihood_ratio_threshold == far.RankingData.ln_likelihood_ratio_threshold
					ranking_data.finish(verbose = self.verbose)
					self.fapfar = far.FAPFAR(ranking_data, coinc_params_distributions.count_above_threshold, threshold = far.RankingData.ln_likelihood_ratio_threshold, livetime = far.get_live_time(seglists))

			# run stream thinca.  update the parameter
			# distribution data from sngls that weren't used in
			# coincs
			for event in self.stream_thinca.add_events(self.coincs_document.xmldoc, self.coincs_document.process_id, events, buf_timestamp, fapfar = self.fapfar):
				self.coinc_params_distributions.add_background(self.coinc_params_distributions.coinc_params((event,), None))
			self.coincs_document.commit()

			# update zero-lag coinc bin counts in
			# coinc_params_distributions.  NOTE:  if likelihood
			# ratios are known then these are the counts of
			# occurances of parameters in coincs above
			# threshold, otherwise they are the counts of
			# occurances of parameters in all coincs.  knowing
			# the meaning of the counts that get recorded is
			# left as an exercise to the user
			if self.stream_thinca.last_coincs:
				for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
					offset_vector = self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)
					if (coinc_event.likelihood >= far.RankingData.ln_likelihood_ratio_threshold or self.marginalized_likelihood_file is None) and not any(offset_vector.values()):
						self.coinc_params_distributions.add_zero_lag(self.coinc_params_distributions.coinc_params(self.stream_thinca.last_coincs.sngl_inspirals(coinc_event_id), offset_vector))

			# Cluster last coincs before recording number of zero
			# lag events or sending alerts to gracedb
			# FIXME Do proper clustering that saves states between
			# thinca intervals and uses an independent clustering
			# window. This can also go wrong if there are multiple
			# events with an identical likelihood.  It will just
			# choose the event with the highest event id
			if self.stream_thinca.last_coincs:
				self.stream_thinca.last_coincs.coinc_event_index = dict([max(self.stream_thinca.last_coincs.coinc_event_index.iteritems(), key = lambda (coinc_event_id, coinc_event): coinc_event.likelihood)])

			# Add events to the observed likelihood histogram post "clustering"
			# FIXME proper clustering is really needed (see above)
			if self.stream_thinca.last_coincs:
				for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
					offset_vector = self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)
					#IFOS come from coinc_inspiral not coinc_event
					ifos = self.stream_thinca.last_coincs.coinc_inspiral_index[coinc_event_id].ifos
					if (coinc_event.likelihood is not None and coinc_event.likelihood >= far.RankingData.ln_likelihood_ratio_threshold) and not any(offset_vector.values()):
						self.ranking_data.zero_lag_likelihood_rates[frozenset(lsctables.instrument_set_from_ifos(ifos))][coinc_event.likelihood,] += 1

			# do GraceDB alerts
			if self.gracedb_far_threshold is not None:
				self.__do_gracedb_alerts()
				self.__update_eye_candy()

	def record_horizon_distance(self, instrument, timestamp, psd, m1, m2, snr_threshold = 8.0):
		with self.lock:
			horizon_distance = reference_psd.horizon_distance(psd, m1 = m1, m2 = m2, snr = snr_threshold, f_min = 10.0, f_max = 0.85 * (psd.f0 + (len(psd.data) - 1) * psd.deltaF))
			assert not (math.isnan(horizon_distance) or math.isinf(horizon_distance))
			# NOTE:  timestamp is cast to float.  should be
			# safe, whitener should be reporting PSDs with
			# integer timestamps.  anyway, we don't need
			# nanosecond precision for the horizon distance
			# history.
			try:
				horizon_history = self.coinc_params_distributions.horizon_history[instrument]
			except KeyError:
				horizon_history = self.coinc_params_distributions.horizon_history[instrument] = far.NearestLeafTree()
			horizon_history[float(timestamp)] = horizon_distance

	def __get_likelihood_file(self):
		# generate a coinc parameter distribution document.  NOTE:
		# likelihood ratio PDFs *are* included if they were present in
		# the --likelihood-file that was loaded.
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, u"gstlal_inspiral", paramdict = {})
		search_summary = ligolw_search_summary.append_search_summary(xmldoc, process, ifos = self.seglistdicts["triggersegments"].keys(), inseg = self.seglistdicts["triggersegments"].extent_all(), outseg = self.seglistdicts["triggersegments"].extent_all())
		# FIXME:  now that we've got all kinds of segment lists
		# being collected, decide which of them should go here.
		far.gen_likelihood_control_doc(xmldoc, process, self.coinc_params_distributions, self.ranking_data, self.seglistdicts["triggersegments"])
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_likelihood_file(self):
		with self.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.__get_likelihood_file(), output, trap_signals = None)
			outstr = output.getvalue()
			output.close()
			return outstr

	def __flush(self):
		# run StreamThinca's .flush().  returns the last remaining
		# non-coincident sngls.  add them to the distribution
		for event in self.stream_thinca.flush(self.coincs_document.xmldoc, self.coincs_document.process_id, fapfar = self.fapfar):
			self.coinc_params_distributions.add_background(self.coinc_params_distributions.coinc_params((event,), None))
		self.coincs_document.commit()

		# update zero-lag bin counts in coinc_params_distributions
		if self.stream_thinca.last_coincs:
			for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
				offset_vector = self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)
				if (coinc_event.likelihood >= far.RankingData.ln_likelihood_ratio_threshold or self.marginalized_likelihood_file is None) and not any(offset_vector.values()):
					self.coinc_params_distributions.add_zero_lag(self.coinc_params_distributions.coinc_params(self.stream_thinca.last_coincs.sngl_inspirals(coinc_event_id), offset_vector))

		# do GraceDB alerts
		if self.gracedb_far_threshold is not None:
			self.__do_gracedb_alerts()

	def flush(self):
		with self.lock:
			self.__flush()

	def __do_gracedb_alerts(self):
		if self.stream_thinca.last_coincs:
			gracedb_client = gracedb.Client()
			gracedb_ids = []
			psdmessage = None
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index

			# This appears to be a silly for loop since
			# coinc_event_index will only have one value, but we're
			# future proofing this at the point where it could have
			# multiple clustered events
			for coinc_event in self.stream_thinca.last_coincs.coinc_event_index.values():
				#
				# continue if the false alarm rate is not low
				# enough, or is nan.
				#

				if coinc_inspiral_index[coinc_event.coinc_event_id].combined_far is None or coinc_inspiral_index[coinc_event.coinc_event_id].combined_far > self.gracedb_far_threshold or numpy.isnan(coinc_inspiral_index[coinc_event.coinc_event_id].combined_far):
					continue

				#
				# retrieve PSDs
				#

				if psdmessage is None:
					if self.verbose:
						print >>sys.stderr, "retrieving PSDs from whiteners and generating psd.xml.gz ..."
					psddict = {}
					for instrument in self.seglistdicts["triggersegments"]:
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

				observatories = "".join(sorted(set(instrument[0] for instrument in self.seglistdicts["triggersegments"])))
				instruments = "".join(sorted(self.seglistdicts["triggersegments"]))
				description = "%s_%s_%s_%s" % (instruments, ("%.4g" % coinc_inspiral_index[coinc_event.coinc_event_id].mass).replace(".", "_").replace("-", "_"), self.gracedb_group, self.gracedb_search)
				end_time = int(coinc_inspiral_index[coinc_event.coinc_event_id].get_end())
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
				xmldoc = self.stream_thinca.last_coincs[coinc_event.coinc_event_id]
				# give the alert all the standard inspiral
				# columns (attributes should all be
				# populated).  FIXME:  ugly.
				sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
				for standard_column in ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id"):
					try:
						sngl_inspiral_table.appendColumn(standard_column)
					except ValueError:
						# already has it
						pass
				ligolw_utils.write_fileobj(xmldoc, message, gz = False, trap_signals = None)
				xmldoc.unlink()
				# FIXME: make this optional from command line?
				if True:
					resp = gracedb_client.createEvent(self.gracedb_group, self.gracedb_pipeline, filename, filecontents = message.getvalue(), search = self.gracedb_search)
					resp_json = resp.json()
					if resp.status != httplib.CREATED:
						print >>sys.stderr, "gracedb upload of %s failed" % filename
					else:
						if self.verbose:
							print >>sys.stderr, "event assigned grace ID %s" % resp_json["graceid"]
						gracedb_ids.append(resp_json["graceid"])
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
					resp = gracedb_client.writeLog(gracedb_id, "strain spectral densities", filename = filename, filecontents = psdmessage.getvalue(), tagname = "psd")
					resp_json = resp.json()
					if resp.status != httplib.CREATED:
						print >>sys.stderr, "gracedb upload of %s for ID %s failed" % (filename, gracedb_id)

	def do_gracedb_alerts(self):
		with self.lock:
			self.__do_gracedb_alerts()

	def __update_eye_candy(self):
		if self.stream_thinca.last_coincs:
			self.ram_history.append((time.time(), (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1048576.)) # GB
			latency_val = None
			snr_val = (0,0)
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index
			for coinc_event_id, coinc_inspiral in coinc_inspiral_index.items():
				# FIXME:  update when a proper column is available
				latency = coinc_inspiral.minimum_duration
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
		with self.lock:
			self.__update_eye_candy()

	def web_get_latency_histogram(self):
		with self.lock:
			for latency, number in zip(self.latency_histogram.centres()[0][1:-1], self.latency_histogram.array[1:-1]):
				yield "%e %e\n" % (latency, number)

	def web_get_latency_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, latency in self.latency_history:
				yield "%f %e\n" % (time, latency)

	def web_get_snr_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.snr_history:
				yield "%f %e\n" % (time, snr)

	def web_get_ram_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, ram in self.ram_history:
				yield "%f %e\n" % (time, ram)

	def web_get_gracedb_far_threshold(self):
		with self.lock:
			if self.gracedb_far_threshold is not None:
				yield "rate=%.17g\n" % self.gracedb_far_threshold
			else:
				yield "rate=\n"

	def web_set_gracedb_far_threshold(self):
		try:
			with self.lock:
				rate = bottle.request.forms["rate"]
				if rate:
					self.gracedb_far_threshold = float(rate)
					yield "OK: rate=%.17g\n" % self.gracedb_far_threshold
				else:
					self.gracedb_far_threshold = None
					yield "OK: rate=\n"
		except:
			yield "error\n"

	def web_get_sngls_snr_threshold(self):
		with self.lock:
			if self.stream_thinca.sngls_snr_threshold is not None:
				yield "snr=%.17g\n" % self.stream_thinca.sngls_snr_threshold
			else:
				yield "snr=\n"

	def web_set_sngls_snr_threshold(self):
		try:
			with self.lock:
				snr_threshold = bottle.request.forms["snr"]
				if snr_threshold:
					self.stream_thinca.sngls_snr_threshold = float(rate)
					yield "OK: snr=%.17g\n" % self.stream_thinca.sngls_snr_threshold
				else:
					self.stream_thinca.sngls_snr_threshold = None
					yield "OK: snr=\n"
		except:
			yield "error\n"

	def __write_output_file(self, filename = None, likelihood_file = None, verbose = False):
		self.__flush()

		# FIXME:  should this be done in .flush() somehow?
		for segtype, seglistdict in self.seglistdicts.items():
			self.coincs_document.llwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID")

		if filename is not None:
			self.coincs_document.filename = filename
		self.coincs_document.write_output_file(verbose = verbose)

		# write the parameter PDF file.  NOTE;  this file contains
		# raw bin counts, and might or might not contain smoothed,
		# normalized, PDF arrays but if it does they will not
		# necessarily correspond to the bin counts. 
		#
		# the parameter PDF arrays cannot be re-computed here
		# because it would interfer with their use by stream
		# thinca.  we want to know precisely when the arrays get
		# updated so we can have a hope of computing the likelihood
		# ratio PDFs correctly.
		if likelihood_file is None:
			likelihood_file = os.path.split(self.coincs_document.filename)
			try: # preserve LIGO-T010150-00 if possible 
				ifo, desc, start, dur = ".".join(likelihood_file[1].split('.')[:-1]).split('-')
				likelihood_file = os.path.join(likelihood_file[0], '%s-%s_SNR_CHI-%s-%s.xml.gz' % (ifo, desc, start, dur))
			except ValueError:
				likelihood_file = os.path.join(likelihood_file[0], '%s_SNR_CHI.xml.gz' % likelihood_file[1].split('.')[0])
		ligolw_utils.write_filename(self.__get_likelihood_file(), likelihood_file, gz = (likelihood_file or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)

		# can't be used anymore
		del self.coincs_document

	def write_output_file(self, filename = None, likelihood_file = None, verbose = False):
		with self.lock:
			self.__write_output_file(filename = filename, likelihood_file = likelihood_file, verbose = verbose)

	def snapshot_output_file(self, description, extension, verbose = False):
		with self.lock:
			coincs_document = self.coincs_document.get_another()
			# We require the likelihood file to have the same name
			# as the input to this program to accumulate statistics
			# as we go 
			self.__write_output_file(filename = self.coincs_document.T050017_filename(description, extension), likelihood_file = self.likelihood_file, verbose = verbose)
			self.coincs_document = coincs_document
