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


import math
import numpy
from scipy import random
import sqlite3
import StringIO
import sys
import time
import httplib
import tempfile

from glue import iterutils
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
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import time_slide as ligolw_time_slide
import lal
from lal import LIGOTimeGPS
from lal import series as lalseries
import ligo.gracedb.rest
from ligo import gracedb


from gstlal import bottle
from gstlal import cbc_template_iir
from gstlal import svd_bank


#
# =============================================================================
#
#                         glue.ligolw Content Handlers
#
# =============================================================================
#


@ligolw_array.use_in
@ligolw_param.use_in
@lsctables.use_in
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def now():
	return LIGOTimeGPS(lal.UTCToGPS(time.gmtime()))


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
			# Write out sngl inspiral table to temp file for
			# trigger generator
			# FIXME teach the trigger generator to get this
			# information a better way
			bank.template_bank_filename = tempfile.NamedTemporaryFile(suffix = ".xml.gz", delete = False).name
			xmldoc = ligolw.Document()
			# FIXME if this table reference is from a DB this
			# is a problem (but it almost certainly isn't)
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
	sngl_inspiral_columns = ("process_id", "ifo", "end_time", "end_time_ns", "eff_distance", "coa_phase", "mass1", "mass2", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "sigmasq", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "template_duration", "event_id", "Gamma0", "Gamma1")

	def __init__(self, url, process_params, comment, instruments, seg, offsetvectors, injection_filename = None, tmp_path = None, replace_file = None, verbose = False):
		#
		# how to make another like us
		#

		self.get_another = lambda: CoincsDocument(url = url, process_params = process_params, comment = comment, instruments = instruments, seg = seg, offsetvectors = offsetvectors, injection_filename = injection_filename, tmp_path = tmp_path, replace_file = replace_file, verbose = verbose)

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
		# insert time slide offset vectors.  remove duplicate
		# offset vectors when done
		#

		time_slide_table = lsctables.TimeSlideTable.get_table(self.xmldoc)
		for offsetvector in offsetvectors:
			time_slide_table.append_offsetvector(offsetvector, self.process)
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

			(self.process.process_id,), = self.connection.cursor().execute("SELECT process_id FROM process WHERE program == ? AND node == ? AND username == ? AND unix_procid == ? AND start_time == ?", (self.process.program, self.process.node, self.process.username, self.process.unix_procid, self.process.start_time)).fetchall()
			self.process.process_id = ilwd.ilwdchar(self.process.process_id)
		else:
			self.connection = self.working_filename = None

		#
		# retrieve references to the table objects, now that we
		# know if they are database-backed or XML objects
		#

		self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(self.xmldoc)


	def commit(self):
		# update output document
		if self.connection is not None:
			self.connection.commit()


	@property
	def process_id(self):
		return self.process.process_id


	def get_next_sngl_id(self):
		return self.sngl_inspiral_table.get_next_id()


	def write_output_url(self, seglistdicts = None, verbose = False):
		if seglistdicts is not None:
			with ligolw_segments.LigolwSegments(self.xmldoc, self.process) as llwsegments:
				for segtype, seglistdict in seglistdicts.items():
					llwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID")

		ligolw_process.set_process_end_time(self.process)

		if self.connection is not None:
			# record the final state of the process row in the
			# database
			cursor = self.connection.cursor()
			cursor.execute("UPDATE process SET end_time = ? WHERE process_id == ?", (self.process.end_time, self.process.process_id))
			cursor.close()
			self.connection.commit()
			dbtables.build_indexes(self.connection, verbose = verbose)
			self.connection.close()
			self.connection = None
			dbtables.put_connection_filename(ligolw_utils.local_path_from_url(self.url), self.working_filename, verbose = verbose)
		else:
			self.sngl_inspiral_table.sort(key = lambda row: (row.end, row.ifo))
			ligolw_utils.write_url(self.xmldoc, self.url, gz = (self.url or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)
		# can no longer be used
		self.xmldoc.unlink()
		del self.xmldoc


#
# =============================================================================
#
#                               GracedB Wrapper
#
# =============================================================================
#


class GracedBWrapper(object):
	retries = 5
	retry_delay = 5.0

	DEFAULT_SERVICE_URL = gracedb.rest.DEFAULT_SERVICE_URL

	def __init__(self, instruments, far_threshold = None, min_instruments = None, group = "Test", search = "LowMass", pipeline = "gstlal", service_url = None, upload_auxiliary_data = True, verbose = False):
		self.instruments = frozenset(instruments)
		self.min_instruments = min_instruments
		self.group = group
		self.search = search
		self.pipeline = pipeline
		self.service_url = service_url if service_url is not None else self.DEFAULT_SERVICE_URL
		self.upload_auxiliary_data = upload_auxiliary_data
		self.verbose = verbose
		# must initialize after .service_url because this might
		# cause the client to be created, which requires
		# .service_url to have already been set
		self.far_threshold = far_threshold

		bottle.route("/gracedb_far_threshold.txt", method = "GET")(self.web_get_gracedb_far_threshold)
		bottle.route("/gracedb_far_threshold.txt", method = "POST")(self.web_set_gracedb_far_threshold)
		bottle.route("/gracedb_min_instruments.txt", method = "GET")(self.web_get_gracedb_min_instruments)
		bottle.route("/gracedb_min_instruments.txt", method = "POST")(self.web_set_gracedb_min_instruments)

	@property
	def far_threshold(self):
		return self._far_threshold

	@far_threshold.setter
	def far_threshold(self, far_threshold):
		self._far_threshold = far_threshold
		if far_threshold is None or far_threshold < 0.:
			self.gracedb_client = None
		else:
			self.gracedb_client = gracedb.rest.GraceDb(self.service_url)

	def web_get_gracedb_far_threshold(self):
		with self.lock:
			if self.far_threshold is not None:
				yield "rate=%.17g\n" % self.far_threshold
			else:
				yield "rate=\n"

	def web_set_gracedb_far_threshold(self):
		try:
			with self.lock:
				rate = bottle.request.forms["rate"]
				if rate:
					self.far_threshold = float(rate)
					yield "OK: rate=%.17g\n" % self.far_threshold
				else:
					self.far_threshold = None
					yield "OK: rate=\n"
		except:
			yield "error\n"

	def web_get_gracedb_min_instruments(self):
		with self.lock:
			if self.min_instruments is not None:
				yield "gracedb_min_instruments=%d\n" % self.min_instruments
			else:
				yield "gracedb_min_instruments=\n"

	def web_set_gracedb_min_instruments(self):
		try:
			with self.lock:
				min_instruments = bottle.request.forms["gracedb_min_instruments"]
				if min_instruments is not None:
					self.min_instruments = int(min_instruments)
					yield "OK: gracedb_min_instruments=%d\n" % self.min_instruments
				else:
					self.min_instruments = None
					yield "OK: gracedb_min_instruments=\n"
		except:
			yield "error\n"

	def __upload_aux_data(self, message, filename, tag, contents, gracedb_ids):
		assert self.gracedb_client is not None, ".gracedb_client is None;  did you forget to set .far_threshold?"
		for gracedb_id in gracedb_ids:
			if self.verbose:
				print >>sys.stderr, "posting '%s' to gracedb ID %d ..." % (filename, gracedb_id)
			for attempt in range(1, self.retries + 1):
				try:
					resp = self.gracedb_client.writeLog(gracedb_id, message, filename = filename, filecontents = contents, tagname = tag)
				except gracedb.rest.HTTPError as resp:
					pass
				else:
					if resp.status == httplib.CREATED:
						break
				print >>sys.stderr, "gracedb upload of %s for ID %s failed on attempt %d/%d: %d: %s"  % (filename, gracedb_id, attempt, self.retries, resp.status, httplib.responses.get(resp.status, "Unknown"))
				time.sleep(random.lognormal(math.log(self.retry_delay), .5))
			else:
				print >>sys.stderr, "gracedb upload of %s for ID %s failed" % (filename, gracedb_id)

	def __upload_aux_xmldoc(self, message, filename, tag, xmldoc, gracedb_ids):
		# check for no-op
		if not gracedb_ids:
			return
		fobj = StringIO.StringIO()
		ligolw_utils.write_fileobj(xmldoc, fobj, gz = filename.endswith(".gz"))
		self.__upload_aux_data(message, filename, tag, fobj.getvalue(), gracedb_ids)
		del fobj

	def do_alerts(self, last_coincs, psddict, rankingstat_xmldoc_func):
		gracedb_ids = []

		# no-op short circuit
		if self.far_threshold is None or not last_coincs:
			return gracedb_ids

		coinc_inspiral_index = last_coincs.coinc_inspiral_index

		# This appears to be a silly for loop since
		# coinc_event_index will only have one value, but we're
		# future proofing this at the point where it could have
		# multiple clustered events
		for coinc_event in last_coincs.coinc_event_index.values():
			#
			# continue if the false alarm rate is not low
			# enough, or is nan, or there aren't enough
			# instruments participating in this coinc
			#

			if coinc_inspiral_index[coinc_event.coinc_event_id].combined_far is None or coinc_inspiral_index[coinc_event.coinc_event_id].combined_far > self.far_threshold or numpy.isnan(coinc_inspiral_index[coinc_event.coinc_event_id].combined_far) or len(last_coincs.sngl_inspirals(coinc_event.coinc_event_id)) < self.min_instruments:
				continue

			#
			# fake a filename for end-user convenience
			#

			observatories = "".join(sorted(set(instrument[0] for instrument in self.instruments)))
			instruments = "".join(sorted(self.instruments))
			description = "%s_%s_%s_%s" % (instruments, ("%.4g" % coinc_inspiral_index[coinc_event.coinc_event_id].mass).replace(".", "_").replace("-", "_"), self.group, self.search)
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
			xmldoc = last_coincs[coinc_event.coinc_event_id]
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
			for event in last_coincs.sngl_inspirals(coinc_event.coinc_event_id):
				snr_time_series = event.snr_time_series
				if snr_time_series is not None:
					xmldoc.childNodes[-1].appendChild(lalseries.build_COMPLEX8TimeSeries(snr_time_series)).appendChild(ligolw_param.Param.from_pyvalue(u"event_id", event.event_id))
			# serialize to XML
			ligolw_utils.write_fileobj(xmldoc, message, gz = False)
			xmldoc.unlink()
			# FIXME: make this optional from command line?
			if True:
				for attempt in range(1, self.retries + 1):
					try:
						resp = self.gracedb_client.createEvent(self.group, self.pipeline, filename, filecontents = message.getvalue(), search = self.search)
					except gracedb.rest.HTTPError as resp:
						pass
					else:
						resp_json = resp.json()
						if resp.status == httplib.CREATED:
							if self.verbose:
								print >>sys.stderr, "event assigned grace ID %s" % resp_json["graceid"]
							gracedb_ids.append(resp_json["graceid"])
							break
					print >>sys.stderr, "gracedb upload of %s failed on attempt %d/%d: %d: %s"  % (filename, attempt, self.retries, resp.status, httplib.responses.get(resp.status, "Unknown"))
					time.sleep(random.lognormal(math.log(self.retry_delay), .5))
				else:
					print >>sys.stderr, "gracedb upload of %s failed" % filename
			else:
				with open(filename, "w") as f:
					f.write(message.getvalue())
			message.close()

		#
		# upload PSDs and ranking statistic data
		#

		if self.upload_auxiliary_data:
			self.__upload_aux_xmldoc("strain spectral densities", "psd.xml.gz", "psd", lalseries.make_psd_xmldoc(psddict), gracedb_ids)
			self.__upload_aux_xmldoc("ranking statistic PDFs", "ranking_data.xml.gz", "ranking statistic", rankingstat_xmldoc_func(), gracedb_ids)

		#
		# done
		#

		return gracedb_ids
