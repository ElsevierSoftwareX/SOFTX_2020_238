
from collections import deque
import threading
import sys
import StringIO
import httplib
import math
import pdb

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
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils

from glue.ligolw.utils import ligolw_sqlite
from glue.ligolw.utils import ligolw_add
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import search_summary as ligolw_search_summary
from glue.ligolw.utils import segments as ligolw_segments

import lal
from pylal.datatypes import LALUnit
from pylal.datatypes import LIGOTimeGPS
from pylal.datatypes import REAL8FrequencySeries
from pylal.xlal.datatypes import postcohinspiraltable

lsctables.LIGOTimeGPS = LIGOTimeGPS

# defined in postcohinspiral_table.h
class PostcohInspiralTable(table.Table):
	tableName = "postcoh:table"
	validcolumns = {
			"end_time":	"int_4s",
			"end_time_ns":	"int_4s",
			"is_background":	"int_4s",
			"livetime":	"int_4s",
			"ifos":		"lstring",
			"pivotal_ifo":	"lstring",
			"tmplt_idx":	"int_4s",
			"pix_idx":	"int_4s",
			"maxsnglsnr":	"real_4",
			"cohsnr":	"real_4",
			"nullsnr":	"real_4",
			"chisq":	"real_4",
			"spearman_pval":	"real_4",
			"fap":		"real_4",
			"far":		"real_4",
			"skymap_fname":	"lstring",
			"template_duration": "real_8",
			"mass1":	"real_4",
			"mass2":	"real_4",
			"mchirp":	"real_4",
			"mtotal":	"real_4",
			"spin1x":	"real_4",
			"spin1y":	"real_4",
			"spin1z":	"real_4",
			"spin2x":	"real_4",
			"spin2y":	"real_4",
			"spin2z":	"real_4",
			"eta":		"real_4",
			"ra":		"real_8",
			"dec":		"real_8"
	}


class PostcohDocument(object):
	def __init__(self, verbose = False):
		self.get_another = lambda: PostcohDocument(verbose = verbose)

		self.filename = None

		#
		# build the XML document
		#

		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())

		# FIXME: process table, search summary table
		# FIXME: should be implemented as lsctables.PostcohInspiralTable
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(PostcohInspiralTable))

	def set_filename(self, filename):
		self.filename = filename

	def write_output_file(self, verbose = False):
		assert self.filename is not None
		ligolw_utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


class Data(object):
	def __init__(self, pipeline, data_output_prefix, cluster_window = 0.5, data_snapshot_interval = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal_spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", verbose = False):
	#
	# initialize
	#
		self.lock = threading.Lock()
		self.verbose = verbose
		self.pipeline = pipeline
		self.is_first_buf = True
		self.is_first_event = True

		# cluster parameters
		self.cluster_window = cluster_window
		self.candidate = None
		self.boundary = None
		self.need_candidate_check = False
		self.cur_event_table = lsctables.New(PostcohInspiralTable)
		self.nevent_clustered = 0

		# gracedb parameters
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url

		self.postcoh_document = PostcohDocument()
		self.postcoh_document_cpy = None
		# this table will go to snapshot file, it stores clustered peaks
		self.postcoh_table = PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

		# snapshot parameters
		self.data_output_prefix = data_output_prefix
		self.data_snapshot_interval = data_snapshot_interval
		self.thread_snapshot = None
		self.t_snapshot_start = None
		self.snapshot_duration = None

	def appsink_new_buffer(self, elem):
		with self.lock:
			buf = elem.emit("pull-buffer")
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			newevents = postcohinspiraltable.PostcohInspiralTable.from_buffer(buf)
			self.need_candidate_check = False

			nevent = len(newevents)
			#print "%f nevent %d" % (buf_timestamp,nevent)
			# initialization
			if self.is_first_buf:
				self.t_snapshot_start = buf_timestamp
				self.is_first_buf = False

			if self.is_first_event and nevent > 0:
				self.boundary = buf_timestamp + self.cluster_window
				self.is_first_event = False

			# extend newevents to cur_event_table
			self.cur_event_table.extend(newevents)

			# the logic of clustering here is quite complicated, fresh up
			# yourself before reading the code
			# check if the newevents is over boundary
			while nevent > 0 and buf_timestamp > self.boundary:
				#if buf_timestamp > 966388199:
					#pdb.set_trace()
				self.cluster(self.cluster_window)

				if self.need_candidate_check:
					self.nevent_clustered += 1
					self.__set_far(self.candidate)
					self.postcoh_table.append(self.candidate)	
					if self.gracedb_far_threshold and self.candidate.far < self.gracedb_far_threshold:
						print self.candidate.end
						print self.candidate.far
						print self.candidate.eta
						#self.__do_gracedb_alerts(self.candidate)
					self.candidate = None
					self.need_candidate_check = False

			# do snapshot when ready
			self.snapshot_duration = buf_timestamp - self.t_snapshot_start
			if self.data_snapshot_interval is not None and self.snapshot_duration >= self.data_snapshot_interval:
				snapshot_filename = self.get_output_filename(self.data_output_prefix, self.t_snapshot_start, self.snapshot_duration)
				self.snapshot_output_file(snapshot_filename)
				self.t_snapshot_start = buf_timestamp

	def __select_head_event(self):
		# last event should have the smallest timestamp
		assert len(self.cur_event_table) != 0
		head_event = self.cur_event_table[0]
		for row in self.cur_event_table:
			if row.end < head_event.end:
				head_event = row	
		return head_event

	def cluster(self, cluster_window):

		# moving window clustering
		if self.candidate is None:
			self.candidate = self.__select_head_event()

		if self.candidate.end > self.boundary:
			self.boundary = self.boundary + cluster_window
			return

		# find the max cohsnr event within the boundary of cur_event_table
		peak_event = self.__select_head_event()
		for row in self.cur_event_table:
			if row.end <= self.boundary and row.cohsnr > peak_event.cohsnr:
				peak_event = row

		if peak_event.end <= self.boundary and peak_event.cohsnr > self.candidate.cohsnr:
			self.candidate = peak_event
			iterutils.inplace_filter(lambda row: row.end > self.boundary, self.cur_event_table)
			# update boundary
			self.boundary = self.candidate.end + cluster_window
			self.need_candidate_check = False
		else:
			iterutils.inplace_filter(lambda row: row.end > self.boundary, self.cur_event_table)
			# update boundary
			self.boundary = self.boundary + cluster_window
			self.need_candidate_check = True

	def __set_far(self, candidate):
		candidate.far = candidate.fap * self.nevent_clustered / candidate.livetime

	def __do_gracedb_alerts(self, trigger):
		# do alerts
		gracedb_client = gracedb.rest.GraceDb(self.gracedb_service_url)
		gracedb_ids = []
		common_messages = []

		sngl_inspiral_columns = ("process_id", "ifo", "end_time", "end_time_ns", "eff_distance", "coa_phase", "mass1", "mass2", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "sigmasq", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id")

		# This appears to be a silly for loop since
		# coinc_event_index will only have one value, but we're
		# future proofing this at the point where it could have
		# multiple clustered events
			
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())

		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = sngl_inspiral_columns))
		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentDefTable, columns = ligolw_segments.LigolwSegmentList.segment_def_columns))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentSumTable, columns = ligolw_segments.LigolwSegmentList.segment_sum_columns))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentTable, columns = ligolw_segments.LigolwSegmentList.segment_columns))

		sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		coinc_def_table = lsctables.CoincDefTable.get_table(xmldoc)
		coinc_table = lsctables.CoincTable.get_table(xmldoc)
		coinc_map_table = lsctables.CoincMapTable.get_table(xmldoc)
		coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(xmldoc)
		time_slide_table = lsctables.TimeSlideTable.get_table(xmldoc)

		#sngl_inspiral_table = self.postcoh_table 
		row = sngl_inspiral_table.RowType()
		for standard_column in ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id"):
		  try:
		    sngl_inspiral_table.appendColumn(standard_column)
		  except ValueError:
		    # already has it
		    pass
		
		# Setting the H1 row
		row.process_id = "process:process_id:10"
		row.ifo = trigger.pivotal_ifo
		row.search = "tmpltbank"
		row.channel = "LDAS-CALIB_STRAIN" 
		row.end_time = trigger.end_time
		row.end_time_ns = trigger.end_time_ns 
		row.end_time_gmst = 0 
		row.impulse_time = 0
		row.impulse_time_ns = 0
		row.template_duration = trigger.template_duration
		row.event_duration = 0
		row.amplitude = 0
		row.eff_distance = 0 
		row.coa_phase = 0
		row.mass1 = trigger.mass1 
		row.mass2 = trigger.mass2
		row.mchirp = trigger.mchirp 
		row.mtotal = trigger.mtotal 
		row.eta = trigger.eta 
		row.kappa = 0 
		row.chi = 1 
		row.tau0 = 0 
		row.tau2 = 0 
		row.tau3 = 0 
		row.tau4 = 0
		row.tau5 = 0 
		row.ttotal = 0 
		row.psi0 = 0 
		row.psi3 = 0 
		row.alpha = 0
		row.alpha1 = 0
		row.alpha2 = 0
		row.alpha3 = 0
		row.alpha4 = 0
		row.alpha5 = 0
		row.alpha6 = 0
		row.beta = 0
		row.f_final = 2048
		row.snr = trigger.maxsnglsnr
		row.chisq = trigger.chisq
		row.chisq_dof = 4
		row.bank_chisq = 0
		row.bank_chisq_dof = 0
		row.cont_chisq = 0
		row.cont_chisq_dof = 0
		row.sigmasq = 0
		row.rsqveto_duration = 0
		row.Gamma0 = 0
		row.Gamma1 = 0
		row.Gamma2 = 0
		row.Gamma3 = 0
		row.Gamma4 = 0
		row.Gamma5 = 0
		row.Gamma6 = 0
		row.Gamma7 = 0
		row.Gamma8 = 0
		row.Gamma9 = 0
		row.spin1x = trigger.spin1x 
		row.spin1y = trigger.spin1y 
		row.spin1z = trigger.spin1z 
		row.spin2x = trigger.spin2x 
		row.spin2y = trigger.spin2y 
		row.spin2z = trigger.spin2z
		row.event_id = "sngl_inspiral:event_id:%d" % self.nevent_clustered
		
		sngl_inspiral_table.append(row)

		if trigger.pivotal_ifo == "H1":
			the_other_ifo = "L1"
		else:
			the_other_ifo = "H1"
		# Setting the the other row
		row.process_id = "process:process_id:10"
		row.ifo = the_other_ifo
		row.search = "tmpltbank"
		row.channel = "LDAS-CALIB_STRAIN" 
		row.end_time = 0
		row.end_time_ns = 0 
		row.end_time_gmst = 0 
		row.impulse_time = 0
		row.impulse_time_ns = 0
		row.template_duration = trigger.template_duration
		row.event_duration = 0
		row.amplitude = 0
		row.eff_distance = 0 
		row.coa_phase = 0
		row.mass1 = trigger.mass1 
		row.mass2 = trigger.mass2 
		row.mchirp = trigger.mchirp 
		row.mtotal = trigger.mtotal 
		row.eta = 0 
		row.kappa = 0 
		row.chi = 1 
		row.tau0 = 0 
		row.tau2 = 0 
		row.tau3 = 0 
		row.tau4 = 0
		row.tau5 = 0 
		row.ttotal = 0 
		row.psi0 = 0 
		row.psi3 = 0 
		row.alpha = 0
		row.alpha1 = 0
		row.alpha2 = 0
		row.alpha3 = 0
		row.alpha4 = 0
		row.alpha5 = 0
		row.alpha6 = 0
		row.beta = 0
		row.f_final = 2048
		row.snr = math.sqrt(math.pow(trigger.cohsnr, 2) - math.pow(trigger.maxsnglsnr, 2))
		row.chisq = 0
		row.chisq_dof = 4
		row.bank_chisq = 0
		row.bank_chisq_dof = 0
		row.cont_chisq = 0
		row.cont_chisq_dof = 0
		row.sigmasq = 0
		row.rsqveto_duration = 0
		row.Gamma0 = 0
		row.Gamma1 = 0
		row.Gamma2 = 0
		row.Gamma3 = 0
		row.Gamma4 = 0
		row.Gamma5 = 0
		row.Gamma6 = 0
		row.Gamma7 = 0
		row.Gamma8 = 0
		row.Gamma9 = 0
		row.spin1x = trigger.spin1x 
		row.spin1y = trigger.spin1y 
		row.spin1z = trigger.spin1z 
		row.spin2x = trigger.spin2x 
		row.spin2y = trigger.spin2y 
		row.spin2z = trigger.spin2z
		row.event_id = "sngl_inspiral:event_id:2"
		
		sngl_inspiral_table.append(row)

		row = coinc_def_table.RowType()
		row.search = "inspiral"
		row.description = "sngl_inspiral<-->sngl_inspiral coincidences"
		row.coinc_def_id = "coinc_definer:coinc_def_id:3"
		row.search_coinc_type = 0
		coinc_def_table.append(row)

		row = coinc_table.RowType()
		row.coinc_event_id = "coinc_event:coinc_event_id:2"
		row.instruments = trigger.ifos
		row.nevents = 2
		row.process_id = "process:process_id:5"
		row.coinc_def_id = "coinc_definer:coinc_def_id:3"
		row.time_slide_id = "time_slide:time_slide_id:6"
		row.likelihood = 0
		coinc_table.append(row)
		
		row = coinc_inspiral_table.RowType()
		row.false_alarm_rate = trigger.fap
		row.mchirp = trigger.mchirp 
		row.minimum_duration = trigger.template_duration
		row.mass = trigger.mtotal
		row.end_time = trigger.end_time
		row.coinc_event_id = "coinc_event:coinc_event_id:2"
		row.snr = trigger.cohsnr
		row.end_time_ns = trigger.end_time_ns
		row.combined_far = trigger.far
		row.ifos = trigger.ifos
		coinc_inspiral_table.append(row)

		#row = coinc_map_table.RowType()
		#pdb.set_trace()
		#row.event_id = "sngl_inspiral:event_id:1"
		#row.event_id = 1
		#row.table_name = "sngl_inspiral"
		#row.coinc_event_id = "coinc_event:coinc_event_id:2"
		#row.coinc_event_id = 2
		#coinc_map_table.append(row)
		#row.event_id = "sngl_insipral:event_id:2"
		#row.event_id = 2
		#row.table_name = "sngl_inspiral"
		#row.coinc_event_id = "coinc_event:coinc_event_id:2"
		#row.coinc_event_id = 2
		#coinc_map_table.append(row)

		row = time_slide_table.RowType()
		row.instrument = "H1"
		row.time_slide_id = "time_slide:time_slide_id:7"
		row.process_id = "process:process_id:10"
		row.offset = 0
		time_slide_table.append(row)
		row.instrument = "L1"
		row.time_slide_id = "time_slide:time_slide_id:7"
		row.process_id = "process:process_id:10"
		row.offset = 0
		time_slide_table.append(row)

		filename = "%s_%s_%d_%d.xml" % (trigger.ifos, trigger.end_time, trigger.tmplt_idx, trigger.pix_idx)
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
		message = StringIO.StringIO()
		#message = file(filename, "w")
		#pdb.set_trace()
		ligolw_utils.write_fileobj(xmldoc, message, gz = False, trap_signals = None)
		xmldoc.unlink()
		
		print >>sys.stderr, "sending %s to gracedb ..." % filename
		# FIXME: make this optional from command line?
		if True:
		  resp = gracedb_client.createEvent(self.gracedb_group, self.gracedb_pipeline, filename, filecontents = message.getvalue(), search = self.gracedb_search)
		  resp_json = resp.json()
		  if resp.status != httplib.CREATED:
		    print >>sys.stderr, "gracedb upload of %s failed" % filename
		  else:
		    print >>sys.stderr, "event assigned grace ID %s" % resp_json["graceid"]
		    gracedb_ids.append(resp_json["graceid"])
		#else:
		#  proc = subprocess.Popen(("/bin/cp", "/dev/stdin", filename), stdin = subprocess.PIPE)
		#  proc.stdin.write(message.getvalue())
		#  proc.stdin.flush()
		#  proc.stdin.close()
		message.close()

		# write a log to explain far
		for gracedb_id in gracedb_ids:
			resp = gracedb_client.writeLog(gracedb_id, "FAR is extrapolated, do not take it too seriously", filename = None, tagname = "analyst_comments")
			if resp.status != httplib.CREATED:
		    		print >>sys.stderr, "gracedb upload of log failed"


	
	def get_output_filename(self, data_output_prefix, t_snapshot_start, snapshot_duration):
		fname = "%s_%d_%d.xml.gz" % (data_output_prefix, t_snapshot_start, snapshot_duration)
		return fname

	def snapshot_output_file(self, filename, verbose = False):
		# FIXME: thread_snapshot must finish before calling this function
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()
		del self.postcoh_document_cpy
		self.postcoh_document_cpy = self.postcoh_document
		self.postcoh_document_cpy.set_filename(filename)
		self.thread_snapshot = threading.Thread(target = self.postcoh_document_cpy.write_output_file, args =(self.postcoh_document_cpy, ))
		self.thread_snapshot.start()
		print "main thread: sub thread return"
		postcoh_document = self.postcoh_document.get_another()
		del self.postcoh_document
		self.postcoh_document = postcoh_document
		print "main thread: snapshot finish"

	def write_output_file(self, filename = None, verbose = False):
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()
		self.__write_output_file(filename)

	def __write_output_file(self, filename = None, verbose = False):
		if filename is not None:
			self.postcoh_document.set_filename(filename)
		self.postcoh_document.write_output_file(verbose = verbose)
