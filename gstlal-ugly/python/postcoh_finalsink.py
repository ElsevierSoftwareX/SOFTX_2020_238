
from collections import deque
import threading
import sys
import StringIO
import httplib
import math
import subprocess
import re
import time
import numpy
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

from gstlal import bottle
from gstlal import reference_psd
from gstlal import postcoh_table_def 

lsctables.LIGOTimeGPS = LIGOTimeGPS

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
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(postcoh_table_def.PostcohInspiralTable))

	def set_filename(self, filename):
		self.filename = filename

	def write_output_file(self, verbose = False):
		assert self.filename is not None
		ligolw_utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)

class OnlinePerformer(object):

	def __init__(self, parent_lock):
		# setup bottle routes
		bottle.route("/latency_history.txt")(self.web_get_latency_history)

		self.latency_history = deque(maxlen = 1000)
		self.parent_lock = parent_lock

	def web_get_latency_history(self):
		with self.parent_lock:
			# first one in the list is sacrificed for a time stamp
			for time, latency in self.latency_history:
				yield "%f %e\n" % (time, latency)

	def update_eye_candy(self, candidate):
		latency_val = (float(candidate.end), float(lal.UTCToGPS(time.gmtime()) - candidate.end))
		self.latency_history.append(latency_val)


class BackgroundStatsUpdater(object):
	def __init__(self, path, input_prefix_list, output, collection_time, ifos):
		self.path = path
		self.input_prefix_list = input_prefix_list
		self.output = output
		self.collection_time = collection_time
		self.ifos = ifos
		self.proc = None

	def update(self, cur_buftime):
		boundary = cur_buftime - self.collection_time
		# list all the files in the path
		nprefix = len(self.input_prefix_list[0].split("_"))
		ls_proc = subprocess.Popen(["ls", self.path], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		ls_out, ls_err = ls_proc.communicate()
		ls_fnames = ls_out.split("\n")
		# find the files within the collection time
		valid_fnames = []
		for ifname in ls_fnames:
			ifname_split = ifname.split("_")
			# FIXME: hard coded the prefix test for the first 4 chars 
			if len(ifname_split) > nprefix and ifname[0:3] == self.input_prefix_list[0][0:3] and ifname[-4:] != "next" and ifname_split[nprefix].isdigit() and int(ifname_split[nprefix]) > boundary:
				valid_fnames.append("%s/%s" % (self.path, ifname))
		if len(valid_fnames) > 0:
			input_for_cmd = ",".join(valid_fnames)
			output_for_cmd = "%s/%s" % (self.path, self.output)
			# execute the cmd in a different process
			cmd = []
			cmd += ["gstlal_cohfar_calc_cdf"]
			cmd += ["--input-filename", input_for_cmd]
			cmd += ["--input-format", "stats"]
			cmd += ["--output-filename", output_for_cmd]
			cmd += ["--ifos", self.ifos]
			print cmd
			self.proc = subprocess.Popen(cmd)



class FinalSink(object):
	def __init__(self, pipeline, need_online_perform, ifos, path, output_prefix, far_factor, cluster_window = 0.5, snapshot_interval = None, cohfar_accumbackground_output_prefix = None, cohfar_assignfap_input_fname = "marginalized_stats", background_collection_time = 86400, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal_spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", verbose = False):
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
		self.cur_event_table = lsctables.New(postcoh_table_def.PostcohInspiralTable)
		self.nevent_clustered = 0

		# gracedb parameters
		self.far_factor = far_factor
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url

		self.postcoh_document = PostcohDocument()
		self.postcoh_document_cpy = None
		# this table will go to snapshot file, it stores clustered peaks
		self.postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

		# snapshot parameters
		self.path = path
		self.output_prefix = output_prefix
		self.snapshot_interval = snapshot_interval
		self.thread_snapshot = None
		self.t_snapshot_start = None
		self.snapshot_duration = None

		# background updater
		self.total_duration = None
		self.t_start = None
		self.background_collection_time = background_collection_time
		self.bsupdater = BackgroundStatsUpdater(path = path, input_prefix_list = cohfar_accumbackground_output_prefix, output = cohfar_assignfap_input_fname, collection_time = background_collection_time, ifos = ifos)

		# online information performer
		self.need_online_perform = need_online_perform
		self.onperformer = OnlinePerformer(parent_lock = self.lock)

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
				self.t_start = buf_timestamp
				self.is_first_buf = False

			if self.is_first_event and nevent > 0:
				self.boundary = buf_timestamp + self.cluster_window
				self.is_first_event = False

			# extend newevents to cur_event_table
			self.cur_event_table.extend(newevents)

			# the logic of clustering here is quite complicated, fresh up
			# yourself before reading the code
			# check if the newevents is over boundary
			while self.boundary and buf_timestamp > self.boundary:
				self.cluster(self.cluster_window)

				if self.need_candidate_check:
					self.nevent_clustered += 1
					self.__set_far(self.candidate)
					self.postcoh_table.append(self.candidate)	
					if self.gracedb_far_threshold and self.candidate.far > 0 and self.candidate.far < self.gracedb_far_threshold:
						self.__do_gracedb_alerts(self.candidate)
					if self.need_online_perform:
						self.onperformer.update_eye_candy(self.candidate)
					self.candidate = None
					self.need_candidate_check = False

			# do snapshot when ready
			self.snapshot_duration = buf_timestamp - self.t_snapshot_start
			self.total_duration = buf_timestamp - self.t_start
			if self.snapshot_interval is not None and self.snapshot_duration >= self.snapshot_interval:
				snapshot_filename = self.get_output_filename(self.output_prefix, self.t_snapshot_start, self.snapshot_duration)
				self.snapshot_output_file(snapshot_filename)
				self.t_snapshot_start = buf_timestamp
				self.bsupdater.update(buf_timestamp)
				self.nevent_clustered = 0

	def __select_head_event(self):
		# last event should have the smallest timestamp
		#assert len(self.cur_event_table) != 0
		if len(self.cur_event_table) == 0:
			return None

		head_event = self.cur_event_table[0]
		for row in self.cur_event_table:
			if row.end < head_event.end:
				head_event = row	
		return head_event

	def cluster(self, cluster_window):

		# moving window clustering
		if self.candidate is None:
			self.candidate = self.__select_head_event()

		if self.candidate is None or self.candidate.end > self.boundary:
			self.boundary = self.boundary + cluster_window
			self.candidate = None
			return

		# find the max cohsnr event within the boundary of cur_event_table
		peak_event = self.__select_head_event()
		for row in self.cur_event_table:
			if row.end <= self.boundary and row.cohsnr > peak_event.cohsnr:
				peak_event = row

		if peak_event is None:
			# no event within the boundary, candidate is the peak, update boundary
			self.boundary = self.boundary + cluster_window
			self.need_candidate_check = True
			return

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
		hack_factor = max(0.5, self.nevent_clustered / float(1 + self.snapshot_duration.gpsSeconds))
		candidate.far = candidate.fap * hack_factor * self.far_factor

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
		row.process_id = "process:process_id:1"
		row.ifo = "L1"
		row.search = self.path
		row.channel = "GDS-CALIB_STRAIN" 
		row.end_time = trigger.end_time_L
		row.end_time_ns = trigger.end_time_ns_L
		row.end_time_gmst = 0 
		row.impulse_time = 0
		row.impulse_time_ns = 0
		row.template_duration = trigger.template_duration
		row.event_duration = 0
		row.amplitude = 0
		row.eff_distance = 0 
		row.coa_phase = trigger.coa_phase_L
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
		row.snr = trigger.snglsnr_L
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
		row.event_id = "sngl_inspiral:event_id:1"
	
		sngl_inspiral_table.append(row)

		row = sngl_inspiral_table.RowType()
		# Setting the the other row
		row.process_id = "process:process_id:1"
		row.ifo = "H1"
		row.search = self.path
		row.channel = "GDS-CALIB_STRAIN" 
		row.end_time = trigger.end_time_H
		row.end_time_ns = trigger.end_time_ns_H 
		row.end_time_gmst = 0 
		row.impulse_time = 0
		row.impulse_time_ns = 0
		row.template_duration = trigger.template_duration
		row.event_duration = 0
		row.amplitude = 0
		row.eff_distance = 0 
		row.coa_phase = trigger.coa_phase_H
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
		row.snr = trigger.snglsnr_H
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
		row.coinc_event_id = "coinc_event:coinc_event_id:1"
		row.instruments = trigger.ifos
		row.nevents = 2
		row.process_id = "process:process_id:1"
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
		row.coinc_event_id = "coinc_event:coinc_event_id:1"
		row.snr = trigger.cohsnr
		row.end_time_ns = trigger.end_time_ns
		row.combined_far = trigger.far
		row.ifos = trigger.ifos
		coinc_inspiral_table.append(row)

		row = coinc_map_table.RowType()
		row.event_id = "sngl_inspiral:event_id:1"
		row.table_name = "sngl_inspiral"
		row.coinc_event_id = "coinc_event:coinc_event_id:1"
		coinc_map_table.append(row)

		row = coinc_map_table.RowType()
		row.event_id = "sngl_insipral:event_id:2"
		row.table_name = "sngl_inspiral"
		row.coinc_event_id = "coinc_event:coinc_event_id:1"
		coinc_map_table.append(row)

		row = time_slide_table.RowType()
		row.instrument = "H1"
		row.time_slide_id = "time_slide:time_slide_id:6"
		row.process_id = "process:process_id:1"
		row.offset = 0
		time_slide_table.append(row)

		row = time_slide_table.RowType()
		row.instrument = "L1"
		row.time_slide_id = "time_slide:time_slide_id:6"
		row.process_id = "process:process_id:1"
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
		gracedb_upload_itrial = 1
		# FIXME: make this optional from cmd line?
		while gracedb_upload_itrial < 10:
			try:
				resp = gracedb_client.createEvent(self.gracedb_group, self.gracedb_pipeline, filename, filecontents = message.getvalue(), search = self.gracedb_search)
				resp_json = resp.json()
				if resp.status != httplib.CREATED:
					print >>sys.stderr, "gracedb upload of %s failed" % filename
				else:
		    			print >>sys.stderr, "event assigned grace ID %s" % resp_json["graceid"]
		    			gracedb_ids.append(resp_json["graceid"])
		    			break
			except:
				gracedb_upload_itrial += 1
		#else:
		#  proc = subprocess.Popen(("/bin/cp", "/dev/stdin", filename), stdin = subprocess.PIPE)
		#  proc.stdin.write(message.getvalue())
		#  proc.stdin.flush()
		#  proc.stdin.close()
		message.close()

		gracedb_upload_itrial = 1
		# write a log to explain far
		#for gracedb_id in gracedb_ids:
		gracedb_id = gracedb_ids[0]
		log_message = "Optimal ra and dec from this coherent pipeline: (%f, %f)" % (trigger.ra, trigger.dec)
		while gracedb_upload_itrial < 10:
			try:
				resp = gracedb_client.writeLog(gracedb_id, log_message , filename = None, tagname = "analyst_comments")
				if resp.status != httplib.CREATED:
		    			print >>sys.stderr, "gracedb upload of log failed"
				else:
					break
			except:
				gracedb_upload_itrial += 1

		if self.verbose:
			print >>sys.stderr, "retrieving PSDs from whiteners and generating psd.xml.gz ..."
		psddict = {}
		instruments = re.findall('..', trigger.ifos)
		for instrument in instruments:
			elem = self.pipeline.get_by_name("lal_whiten_%s" % instrument)
			# FIXME:  remove
			# LIGOTimeGPS type cast
			# when we port to swig
			# version of
			# REAL8FrequencySeries
			psddict[instrument] = REAL8FrequencySeries(
				name = "PSD",
				epoch = LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0),
				f0 = 0.0,
				deltaF = elem.get_property("delta-f"),
				sampleUnits = LALUnit("s strain^2"),	# FIXME:  don't hard-code this
				data = numpy.array(elem.get_property("mean-psd"))
				)
		fobj = StringIO.StringIO()
		reference_psd.write_psd_fileobj(fobj, psddict, gz = True, trap_signals = None)
		common_messages.append(("strain spectral densities", "psd.xml.gz", "psd", fobj.getvalue()))


		#
		# do PSD and ranking data file uploads
		#

		while common_messages:
			message, filename, tag, contents = common_messages.pop()
			gracedb_upload_itrial = 1
			gracedb_id = gracedb_ids[0]
			while gracedb_upload_itrial < 10:
				try:
					resp = gracedb_client.writeLog(gracedb_id, message, filename = filename, filecontents = contents, tagname = tag)
					resp_json = resp.json()
					if resp.status != httplib.CREATED:
						print >>sys.stderr, "gracedb upload of %s for ID %s failed" % (filename, gracedb_id)
					else:
						break
				except:
					gracedb_upload_itrial += 1


	
	def get_output_filename(self, output_prefix, t_snapshot_start, snapshot_duration):
		fname = "%s/%s_%d_%d.xml.gz" % (self.path, output_prefix, t_snapshot_start, snapshot_duration)
		return fname

	def snapshot_output_file(self, filename, verbose = False):
		self.__check_internal_process_finish()
		# free the memory
		del self.postcoh_document_cpy
		self.postcoh_document_cpy = self.postcoh_document
		self.postcoh_document_cpy.set_filename(filename)
		self.thread_snapshot = threading.Thread(target = self.postcoh_document_cpy.write_output_file, args =(self.postcoh_document_cpy, ))
		self.thread_snapshot.start()

		# set a new document for postcoh_document
		postcoh_document = self.postcoh_document.get_another()
		self.postcoh_document = postcoh_document
		del self.postcoh_table
		self.postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

	def __check_internal_process_finish(self):
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()
	
		if self.bsupdater.proc is not None and self.bsupdater.proc.poll() is None:
			tmp_out, tmp_err = self.bsupdater.proc.communicate()

	def write_output_file(self, filename = None, verbose = False):
		self.__check_internal_process_finish()
		self.__write_output_file(filename)

	def __write_output_file(self, filename = None, verbose = False):
		if filename is not None:
			self.postcoh_document.set_filename(filename)
		self.postcoh_document.write_output_file(verbose = verbose)
