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

## @file
# The python module to implement things needed by gstlal_inspiral
#
# ### Review Status
#
# STATUS: reviewed with actions
#
# | Names                                          | Hash                                        | Date        | Diff to Head of Master     |
# | -------------------------------------------    | ------------------------------------------- | ----------  | -------- |
# | Kipp Cannon, Chad Hanna, Jolien Creighton, Florent Robinet, B. Sathyaprakash, Duncan Meacher, T.G.G. Li    | b8fef70a6bafa52e3e120a495ad0db22007caa20 | 2014-12-03 | <a href="@gstlal_inspiral_cgit_diff/python/inspiral.py?id=HEAD&id2=b8fef70a6bafa52e3e120a495ad0db22007caa20">inspiral.py</a> |
# | Kipp Cannon, Chad Hanna, Jolien Creighton, B. Sathyaprakash, Duncan Meacher                                | 72875f5cb241e8d297cd9b3f9fe309a6cfe3f716 | 2015-11-06 | <a href="@gstlal_inspiral_cgit_diff/python/inspiral.py?id=HEAD&id2=72875f5cb241e8d297cd9b3f9fe309a6cfe3f716">inspiral.py</a> |
#
# #### Action items
# - Document examples of how to get SNR history, etc., to a web browser in an offline search
# - Long term goal: Using template duration (rather than chirp mass) should load balance the pipeline and improve statistics
# - L651: One thing to sort out is the signal probability while computing coincs
# - L640-L647: Get rid of obsolete comments
# - L667: Make sure timeslide events are not sent to GRACEDB
# - Lxxx: Can normalisation of the tail of the distribution pre-computed using fake data?
# - L681: fmin should not be hard-coded to 10 Hz. horizon_distance will be horribly wrong if psd is constructed, e.g. using some high-pass filter. For example, change the default to 40 Hz.
# - L817: If gracedb upload failed then it should be possible to identify the failure, the specifics of the trigger that encountered failure and a way of submitting the trigger again to gracedb is important. Think about how to clean-up failures.
# - Mimick gracedb upload failures and see if the code crashes


## @package inspiral

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from collections import deque
import itertools
import math
import numpy
import os
import resource
from scipy import random
import sqlite3
import StringIO
import subprocess
import sys
import threading
import time
import httplib
import tempfile
import shutil

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

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
import lal
from lal import LIGOTimeGPS
from lal import series as lalseries
from pylal import rate

from gstlal import bottle
from gstlal import reference_psd
from gstlal import streamthinca
from gstlal import svd_bank
from gstlal import cbc_template_iir
from gstlal import far


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def message_new_checkpoint(src, timestamp = None):
	s = Gst.Structure.new_empty("CHECKPOINT")
	message = Gst.Message.new_application(src, s)
	if timestamp is not None:
		message.timestamp = timestamp
	return message


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


def subdir_from_T050017_filename(fname):
	path = str(fname.split("-")[2])[:5]
	try:
		os.mkdir(path)
	except OSError:
		pass
	return path


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

	def __init__(self, url, process_params, comment, instruments, seg, injection_filename = None, time_slide_file = None, tmp_path = None, replace_file = None, verbose = False):
		#
		# how to make another like us
		#

		self.get_another = lambda: CoincsDocument(url = url, process_params = process_params, comment = comment, instruments = instruments, seg = seg, injection_filename = injection_filename, time_slide_file = time_slide_file, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

		#
		# url
		#

		self.url = url

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

		if url is not None and url.endswith('.sqlite'):
			self.working_filename = dbtables.get_connection_filename(ligolw_utils.local_path_from_url(url), tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)
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
		return self.search_summary.out_segment


	def add_to_search_summary_outseg(self, seg):
		out_segs = segments.segmentlist([self.search_summary_outseg])
		if out_segs == [None]:
			# out segment not yet initialized
			del out_segs[:]
		out_segs |= segments.segmentlist([seg])
		self.search_summary.out_segment = out_segs.extent()


	def get_next_sngl_id(self):
		return self.sngl_inspiral_table.get_next_id()


	def T050017_filename(self, description, extension):
		start, end = self.search_summary_outseg
		start, end = int(math.floor(start)), int(math.ceil(end))
		return "%s-%s-%d-%d.%s" % ("".join(sorted(self.process.instruments)), description, start, end - start, extension)


	def write_output_url(self, verbose = False):
		self.llwsegments.finalize()
		ligolw_process.set_process_end_time(self.process)

		if self.connection is not None:
			seg = self.search_summary_outseg
			# record the final state of the search_summary and
			# process rows in the database
			cursor = self.connection.cursor()
			if seg is not None:
				cursor.execute("UPDATE search_summary SET out_start_time = ?, out_start_time_ns = ?, out_end_time = ?, out_end_time_ns = ? WHERE process_id == ?", (seg[0].gpsSeconds, seg[0].gpsNanoSeconds, seg[1].gpsSeconds, seg[1].gpsNanoSeconds, self.search_summary.process_id))
			cursor.execute("UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral) WHERE process_id == ?", (self.search_summary.process_id,))
			cursor.execute("UPDATE process SET end_time = ? WHERE process_id == ?", (self.process.end_time, self.process.process_id))
			cursor.close()
			self.connection.commit()
			dbtables.build_indexes(self.connection, verbose = verbose)
			self.connection.close()
			self.connection = None
			dbtables.put_connection_filename(ligolw_utils.local_path_from_url(self.url), self.working_filename, verbose = verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			self.search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_utils.write_url(self.xmldoc, self.url, gz = (self.url or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


class Data(object):
	def __init__(self, url, process_params, pipeline, seg, coinc_params_distributions, zero_lag_ranking_stats = None, marginalized_likelihood_file = None, likelihood_url_namedtuple = None, injection_filename = None, time_slide_file = None, comment = None, tmp_path = None, likelihood_snapshot_interval = None, thinca_interval = 50.0, min_log_L = None, sngls_snr_threshold = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal", gracedb_service_url = "https://gracedb.ligo.org/api/", replace_file = True, upload_auxiliary_data_to_gracedb = True, verbose = False):
		#
		# initialize
		#

		self.lock = threading.Lock()
		self.pipeline = pipeline
		self.verbose = verbose
		self.upload_auxiliary_data_to_gracedb = upload_auxiliary_data_to_gracedb
		# None to disable likelihood ratio assignment, otherwise a filename
		self.marginalized_likelihood_file = marginalized_likelihood_file
		self.likelihood_url_namedtuple = likelihood_url_namedtuple
		# None to disable periodic snapshots, otherwise seconds
		# set to 1.0 to disable background data decay
		self.likelihood_snapshot_interval = likelihood_snapshot_interval
		self.likelihood_snapshot_timestamp = None
		# gracedb far threshold
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url

		#
		# setup bottle routes
		#

		bottle.route("/latency_histogram.txt")(self.web_get_latency_histogram)
		bottle.route("/latency_history.txt")(self.web_get_latency_history)
		bottle.route("/snr_history.txt")(self.web_get_snr_history)
		bottle.route("/ram_history.txt")(self.web_get_ram_history)
		bottle.route("/likelihood.xml")(self.web_get_likelihood_file)
		bottle.route("/zero_lag_ranking_stats.xml")(self.web_get_zero_lag_ranking_stats_file)
		bottle.route("/gracedb_far_threshold.txt", method = "GET")(self.web_get_gracedb_far_threshold)
		bottle.route("/gracedb_far_threshold.txt", method = "POST")(self.web_set_gracedb_far_threshold)
		bottle.route("/sngls_snr_threshold.txt", method = "GET")(self.web_get_sngls_snr_threshold)
		bottle.route("/sngls_snr_threshold.txt", method = "POST")(self.web_set_sngls_snr_threshold)

		#
		# initialize document to hold coincs and segments
		#

		self.coincs_document = CoincsDocument(url, process_params, comment, coinc_params_distributions.instruments, seg, injection_filename = injection_filename, time_slide_file = time_slide_file, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

		#
		# attach a StreamThinca instance to ourselves
		#

		self.stream_thinca = streamthinca.StreamThinca(
			coincidence_threshold = coinc_params_distributions.delta_t,
			thinca_interval = thinca_interval,	# seconds
			min_instruments = coinc_params_distributions.min_instruments,
			min_log_L = min_log_L,
			sngls_snr_threshold = sngls_snr_threshold
		)

		#
		# setup likelihood ratio book-keeping.  seglistsdicts to be
		# populated the pipeline handler.  zero_lag_ranking_stats
		# is a RankingData object that is used to accumulate a
		# histogram of the likelihood ratio values assigned to
		# zero-lag candidates.  this is required to implement the
		# extinction model for low-significance events during
		# online running but otherwise is optional.  ranking_data
		# contains the RankingData object used to initialize the
		# FAPFAR object for on-the-fly FAP and FAR assignment.
		# except to initialize the FAPFAR object it is not used for
		# anything, but is retained so that it can be exposed
		# through the web interface for diagnostic purposed and
		# uploaded to gracedb with candidates.
		#

		self.coinc_params_distributions = coinc_params_distributions
		self.ranking_data = None
		self.seglistdicts = None
		self.zero_lag_ranking_stats = zero_lag_ranking_stats
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
			buf = elem.emit("pull-sample").get_buffer()
			result, mapinfo = buf.map(Gst.MapFlags.READ)
			assert result
			# NOTE NOTE NOTE NOTE
			# It is critical that the correct class'
			# .from_buffer() method be used here.  This code is
			# interpreting the buffer's contents as an array of
			# C structures and building instances of python
			# wrappers of those structures but if the python
			# wrappers are for the wrong structure declaration
			# then terrible terrible things will happen
			# NOTE NOTE NOTE NOTE
			# FIXME why does mapinfo.data come out as an empty list on some occasions???
			if mapinfo.data:
				events = streamthinca.SnglInspiral.from_buffer(mapinfo.data)
			else:
				events = []
			buf.unmap(mapinfo)
			# FIXME:  ugly way to get the instrument
			instrument = elem.get_name().split("_", 1)[0]

			# update search_summary out segment and our
			# livetime
			buf_timestamp = LIGOTimeGPS(0, buf.pts)
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

				# if a reference likelihood file is given,
				# overwrite coinc_params_distributions with its
				# contents.  in either case, invoke .finish()
				# to re-populate the PDF arrays from the raw
				# counts
				# FIXME There is currently no guarantee that
				# the reference_likelihood_file on disk will
				# have updated since the last snapshot, but for
				# our purpose it should not have that large of
				# an effect. The data loaded should never be
				# older than the snapshot before last
				if self.likelihood_url_namedtuple.reference_likelihood_url is not None:
					params_before = self.coinc_params_distributions.instruments, self.coinc_params_distributions.min_instruments, self.coinc_params_distributions.delta_t
					self.coinc_params_distributions, _, seglists = far.parse_likelihood_control_doc(ligolw_utils.load_url(self.likelihood_url_namedtuple.reference_likelihood_url, verbose = self.verbose, contenthandler = far.ThincaCoincParamsDistributions.LIGOLWContentHandler))
					params_after  = self.coinc_params_distributions.instruments, self.coinc_params_distributions.min_instruments, self.coinc_params_distributions.delta_t
					if params_before != params_after:
						raise ValueError("'%s' contains incompatible ranking statistic configuration" % self.likelihood_url_namedtuple.reference_likelihood_url)
					self.coinc_params_distributions.finish(segs = seglists, verbose = self.verbose)
				else:
					self.coinc_params_distributions.finish(segs = self.seglistdicts["triggersegments"], verbose = self.verbose)

				# post a checkpoint message.
				# FIXME:  make sure this triggers
				# self.snapshot_output_url() to be
				# invoked.  lloidparts takes care of that
				# for now, but spreading the program logic
				# around like that isn't a good idea, this
				# code should be responsible for it
				# somehow, no?  NOTE:
				# self.snapshot_output_url() does not
				# write the coinc_params_distributions
				# object to disk if a reference likelihood
				# file is given, so the the thing that was
				# just read in is not written back out
				# again.  see the comment in that function
				# about turning it into two handlers and
				# only hooking up which ones are needed.
				self.pipeline.get_bus().post(message_new_checkpoint(self.pipeline, timestamp = buf_timestamp.ns()))

				if self.marginalized_likelihood_file is not None:
					# enable streamthinca's likelihood
					# ratio assignment using our own,
					# local, parameter distribution
					# data
					self.stream_thinca.coinc_params_distributions = self.coinc_params_distributions

					# read the marginalized likelihood
					# ratio distributions that have
					# been updated asynchronously and
					# initialize a FAP/FAR assignment
					# machine from it.
					self.ranking_data, seglists = far.parse_likelihood_control_doc(ligolw_utils.load_filename(self.marginalized_likelihood_file, verbose = self.verbose, contenthandler = far.ThincaCoincParamsDistributions.LIGOLWContentHandler))[1:]
					if self.ranking_data is None:
						raise ValueError("\"%s\" does not contain ranking statistic PDFs" % self.marginalized_likelihood_file)
					self.ranking_data = self.ranking_data.new_with_extinction(self.ranking_data)[0]
					self.ranking_data.finish(verbose = self.verbose)
					self.fapfar = far.FAPFAR(self.ranking_data, livetime = far.get_live_time(seglists))

			# run stream thinca.  update the parameter
			# distribution data from sngls that weren't used in
			# coincs.  NOTE:  we rely on the arguments to
			# .chain() being evaluated in left-to-right order
			# so that .add_events() is evaluated before
			# .last_coincs because the former initializes the
			# latter.  we skip singles collected during times
			# when only one instrument was on.  NOTE:  the
			# definition of "on" is blurry since we can recover
			# triggers with end times outside of the available
			# strain data, but the purpose of this test is
			# simply to prevent signals occuring during
			# single-detector times from contaminating our
			# noise model, so it's not necessary for this test
			# to be super precisely defined.
			# FIXME FIXME FIXME buf_timestamp - 1.0 is used to
			# be the maximum offset a template can have within
			# a buffer that would cause its end time to be
			# before the true buffer boundary start.  This
			# comes from the largest negative offset in any
			# given SVD bank.  ITAC should be patched to do
			# this once synchronization issues are sorted
			for event in itertools.chain(self.stream_thinca.add_events(self.coincs_document.xmldoc, self.coincs_document.process_id, events, buf_timestamp - 1.0, fapfar = self.fapfar), self.stream_thinca.last_coincs.single_sngl_inspirals() if self.stream_thinca.last_coincs else ()):

				if len(self.seglistdicts["whitehtsegments"].keys_at(event.end)) > 1:
					self.coinc_params_distributions.add_background(self.coinc_params_distributions.coinc_params((event,), None, mode = "counting"))
			self.coincs_document.commit()

			# update zero-lag coinc bin counts in
			# coinc_params_distributions.
			if self.stream_thinca.last_coincs:
				for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
					if coinc_event.time_slide_id in self.stream_thinca.last_coincs.zero_lag_time_slide_ids:
						self.coinc_params_distributions.add_zero_lag(self.coinc_params_distributions.coinc_params(self.stream_thinca.last_coincs.sngl_inspirals(coinc_event_id), self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)))

			# Cluster last coincs before recording number of zero
			# lag events or sending alerts to gracedb
			# FIXME Do proper clustering that saves states between
			# thinca intervals and uses an independent clustering
			# window. This can also go wrong if there are multiple
			# events with an identical likelihood.  It will just
			# choose the event with the highest event id
			if self.stream_thinca.last_coincs:
				self.stream_thinca.last_coincs.coinc_event_index = dict([max(self.stream_thinca.last_coincs.coinc_event_index.iteritems(), key = lambda (coinc_event_id, coinc_event): coinc_event.likelihood)])

			# Add events to the observed likelihood histogram
			# post "clustering"
			# FIXME proper clustering is really needed (see
			# above)
			if self.stream_thinca.last_coincs and self.zero_lag_ranking_stats is not None:
				for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
					if coinc_event.likelihood is not None and coinc_event.time_slide_id in self.stream_thinca.last_coincs.zero_lag_time_slide_ids:
						self.zero_lag_ranking_stats.zero_lag_likelihood_rates[frozenset(self.stream_thinca.last_coincs.coinc_inspiral_index[coinc_event_id].instruments)][coinc_event.likelihood,] += 1

			# do GraceDB alerts
			if self.gracedb_far_threshold is not None:
				self.__do_gracedb_alerts()
				self.__update_eye_candy()

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
			ligolw_utils.write_fileobj(self.__get_likelihood_file(), output)
			outstr = output.getvalue()
			output.close()
			return outstr

	def __get_zero_lag_ranking_stats_file(self):
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, u"gstlal_inspiral", paramdict = {})
		search_summary = ligolw_search_summary.append_search_summary(xmldoc, process, ifos = self.seglistdicts["triggersegments"].keys(), inseg = self.seglistdicts["triggersegments"].extent_all(), outseg = self.seglistdicts["triggersegments"].extent_all())
		# FIXME:  now that we've got all kinds of segment lists
		# being collected, decide which of them should go here.
		far.gen_likelihood_control_doc(xmldoc, process, None, self.zero_lag_ranking_stats, self.seglistdicts["triggersegments"])
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_zero_lag_ranking_stats_file(self):
		with self.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.__get_zero_lag_ranking_stats_file(), output)
			outstr = output.getvalue()
			output.close()
			return outstr

	def __flush(self):
		# run StreamThinca's .flush().  returns the last remaining
		# non-coincident sngls.  add them to the distribution
		for event in self.stream_thinca.flush(self.coincs_document.xmldoc, self.coincs_document.process_id, fapfar = self.fapfar):
			self.coinc_params_distributions.add_background(self.coinc_params_distributions.coinc_params((event,), None, mode = "counting"))
		self.coincs_document.commit()

		# update zero-lag bin counts in coinc_params_distributions
		if self.stream_thinca.last_coincs:
			for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
				offset_vector = self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)
				if not any(offset_vector.values()):
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

		# Add events to the observed likelihood histogram post
		# "clustering"
		# FIXME proper clustering is really needed (see above)
		if self.stream_thinca.last_coincs and self.zero_lag_ranking_stats is not None:
			for coinc_event_id, coinc_event in self.stream_thinca.last_coincs.coinc_event_index.items():
				offset_vector = self.stream_thinca.last_coincs.offset_vector(coinc_event.time_slide_id)
				instruments = frozenset(self.stream_thinca.last_coincs.coinc_inspiral_index[coinc_event_id].instruments)
				if coinc_event.likelihood is not None and not any(offset_vector.values()):
					self.zero_lag_ranking_stats.zero_lag_likelihood_rates[instruments][coinc_event.likelihood,] += 1


		# do GraceDB alerts
		if self.gracedb_far_threshold is not None:
			self.__do_gracedb_alerts()

	def flush(self):
		with self.lock:
			self.__flush()

	def __do_gracedb_alerts(self, retries = 5, retry_delay = 5.):
		# no-op short circuit
		if not self.stream_thinca.last_coincs:
			return

		gracedb_client = gracedb.rest.GraceDb(self.gracedb_service_url)
		gracedb_ids = []
		common_messages = []
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
			# retrieve PSDs and ranking data
			#

			if not common_messages and self.upload_auxiliary_data_to_gracedb:
				if self.verbose:
					print >>sys.stderr, "retrieving PSDs from whiteners and generating psd.xml.gz ..."
				psddict = {}
				for instrument in self.seglistdicts["triggersegments"]:
					elem = self.pipeline.get_by_name("lal_whiten_%s" % instrument)
					data = numpy.array(elem.get_property("mean-psd"))
					psddict[instrument] = lal.CreateREAL8FrequencySeries(
						name = "PSD",
						epoch = LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0),
						f0 = 0.0,
						deltaF = elem.get_property("delta-f"),
						sampleUnits = lal.Unit("s strain^2"),	# FIXME:  don't hard-code this
						length = len(data)
					)
					psddict[instrument].data.data = data
				fobj = StringIO.StringIO()
				reference_psd.write_psd_fileobj(fobj, psddict, gz = True)
				common_messages.append(("strain spectral densities", "psd.xml.gz", "psd", fobj.getvalue()))

				if self.verbose:
					print >>sys.stderr, "generating ranking_data.xml.gz ..."
				fobj = StringIO.StringIO()
				ligolw_utils.write_fileobj(self.__get_likelihood_file(), fobj, gz = True)
				common_messages.append(("ranking statistic PDFs", "ranking_data.xml.gz", "ranking statistic", fobj.getvalue()))
				del fobj

			#
			# fake a filename for end-user convenience
			#

			observatories = "".join(sorted(set(instrument[0] for instrument in self.seglistdicts["triggersegments"])))
			instruments = "".join(sorted(self.seglistdicts["triggersegments"]))
			description = "%s_%s_%s_%s" % (instruments, ("%.4g" % coinc_inspiral_index[coinc_event.coinc_event_id].mass).replace(".", "_").replace("-", "_"), self.gracedb_group, self.gracedb_search)
			end_time = int(coinc_inspiral_index[coinc_event.coinc_event_id].end)
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
			# add SNR time series if available
			for event in self.stream_thinca.last_coincs.sngl_inspirals(coinc_event.coinc_event_id):
				snr_time_series = event.snr_time_series
				if snr_time_series is not None:
					snr_time_series = xmldoc.childNodes[-1].appendChild(lalseries.build_COMPLEX8TimeSeries(snr_time_series))
					snr_time_series.appendChild(ligolw_param.Param.from_pyvalue(u"event_id", event.event_id))
			# serialize to XML
			ligolw_utils.write_fileobj(xmldoc, message, gz = False)
			xmldoc.unlink()
			# FIXME: make this optional from command line?
			if True:
				for attempt in range(1, retries + 1):
					try:
						resp = gracedb_client.createEvent(self.gracedb_group, self.gracedb_pipeline, filename, filecontents = message.getvalue(), search = self.gracedb_search)
					except gracedb.rest.HTTPError as resp:
						pass
					else:
						resp_json = resp.json()
						if resp.status == httplib.CREATED:
							if self.verbose:
								print >>sys.stderr, "event assigned grace ID %s" % resp_json["graceid"]
							gracedb_ids.append(resp_json["graceid"])
							break
					print >>sys.stderr, "gracedb upload of %s failed on attempt %d/%d: %d: %s"  % (filename, attempt, retries, resp.status, httplib.responses.get(resp.status, "Unknown"))
					time.sleep(random.lognormal(math.log(retry_delay), .5))
				else:
					print >>sys.stderr, "gracedb upload of %s failed" % filename
			else:
				proc = subprocess.Popen(("/bin/cp", "/dev/stdin", filename), stdin = subprocess.PIPE)
				proc.stdin.write(message.getvalue())
				proc.stdin.flush()
				proc.stdin.close()
			message.close()

		#
		# do PSD and ranking data file uploads
		#

		while common_messages:
			message, filename, tag, contents = common_messages.pop()
			for gracedb_id in gracedb_ids:
				for attempt in range(1, retries + 1):
					try:
						resp = gracedb_client.writeLog(gracedb_id, message, filename = filename, filecontents = contents, tagname = tag)
					except gracedb.rest.HTTPError as resp:
						pass
					else:
						if resp.status == httplib.CREATED:
							break
					print >>sys.stderr, "gracedb upload of %s for ID %s failed on attempt %d/%d: %d: %s"  % (filename, gracedb_id, attempt, retries, resp.status, httplib.responses.get(resp.status, "Unknown"))
					time.sleep(random.lognormal(math.log(retry_delay), .5))
				else:
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
					t = float(coinc_inspiral_index[coinc_event_id].end)
					latency_val = (t, latency)
				snr = coinc_inspiral_index[coinc_event_id].snr
				if snr >= snr_val[1]:
					t = float(coinc_inspiral_index[coinc_event_id].end)
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
					self.stream_thinca.sngls_snr_threshold = float(snr_threshold)
					yield "OK: snr=%.17g\n" % self.stream_thinca.sngls_snr_threshold
				else:
					self.stream_thinca.sngls_snr_threshold = None
					yield "OK: snr=\n"
		except:
			yield "error\n"

	def __write_output_url(self, url = None, verbose = False):
		self.__flush()

		# FIXME:  should this be done in .flush() somehow?
		for segtype, seglistdict in self.seglistdicts.items():
			self.coincs_document.llwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID")

		if url is not None:
			self.coincs_document.url = url
		self.coincs_document.write_output_url(verbose = verbose)

	def __write_likelihood_url(self, url, description, snapshot = False, verbose = False):
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
		#
		# FIXME Can we delete the temporary file atomic process now
		# that glue creates temp files before writing

		scheme_host_path, filename = os.path.split(url)
		tmp_likelihood_url = os.path.join(scheme_host_path, 'tmp_%s' % filename)
		ligolw_utils.write_url(self.__get_likelihood_file(), tmp_likelihood_url, gz = (tmp_likelihood_url or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)
		shutil.move(ligolw_utils.local_path_from_url(tmp_likelihood_url), ligolw_utils.local_path_from_url(url))
		# Snapshots get their own custom file and path
		if snapshot:
			fname = self.coincs_document.T050017_filename(description + '_DISTSTATS', 'xml.gz')
			shutil.copy(ligolw_utils.local_path_from_url(url), os.path.join(subdir_from_T050017_filename(fname), fname))

	def __write_zero_lag_ranking_stats_file(self, filename, verbose = False):
		ligolw_utils.write_filename(self.__get_zero_lag_ranking_stats_file(), filename, gz = (filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)

	def write_output_url(self, url = None, description = "", verbose = False):
		with self.lock:
			self.__write_output_url(url = url, verbose = verbose)
			if self.likelihood_url_namedtuple.likelihood_url: 
				self.__write_likelihood_url(self.likelihood_url_namedtuple.likelihood_url, description, verbose = verbose)

			# can't be used anymore
			del self.coincs_document

	def snapshot_output_url(self, description, extension, zero_lag_ranking_stats_filename = None, verbose = False):
		with self.lock:
			coincs_document = self.coincs_document.get_another()
			# We require the likelihood file to have the same name
			# as the input to this program to accumulate statistics
			# as we go
			fname = self.coincs_document.T050017_filename(description, extension)
			fname = os.path.join(subdir_from_T050017_filename(fname), fname)
			self.__write_output_url(url = fname, verbose = verbose)
			if self.likelihood_url_namedtuple.likelihood_url:
				self.__write_likelihood_url(self.likelihood_url_namedtuple.likelihood_url, description, snapshot = True, verbose = verbose)
			if zero_lag_ranking_stats_filename is not None:
				self.__write_zero_lag_ranking_stats_file(zero_lag_ranking_stats_filename, verbose = verbose)

			# can't be used anymore
			del self.coincs_document
			self.coincs_document = coincs_document
