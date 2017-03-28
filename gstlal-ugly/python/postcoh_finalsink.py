#
# Copyright (C) 2015 Qi Chu
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
import os
import fcntl
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
	def __init__(self, path, input_prefix_list, ifos, output_list_string = None, collection_time_string = None):
		self.path = path
		self.input_prefix_list = input_prefix_list
		self.ifos = ifos
		self.procs = []
		if output_list_string is not None:
			self.output = output_list_string.split(",")

		self.collection_time = []
		if collection_time_string is not None:
			times = collection_time_string.split(",")
			for itime in times:
				self.collection_time.append(int(itime))


	def __wait_last_process_finish(self):
		if self.procs is not None:
			for proc in self.procs:
				if proc.poll() is None:
					proc.wait()


	def update_fap_stats(self, cur_buftime):
		self.__wait_last_process_finish()
		# list all the files in the path
		#nprefix = len(self.input_prefix_list[0].split("_"))
		ls_proc = subprocess.Popen(["ls", self.path], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		ls_out = ""
		try:
			ls_out = subprocess.check_output(["grep", "stats"], stdin = ls_proc.stdout)
		except:
			print "no stats file yet"
			return
		ls_proc.wait()
		ls_fnames = ls_out.split("\n")
		#pdb.set_trace()	
		for (i, collection_time) in enumerate(self.collection_time):
			boundary = cur_buftime - collection_time
			# find the files within the collection time
			valid_fnames = []
			for ifname in ls_fnames:
				ifname_split = ifname.split("_")
				# FIXME: hard coded the prefix test for the first 4 chars: must be bank
				if len(ifname_split)> 1 and ifname[-4:] != "next" and ifname_split[-2].isdigit() and int(ifname_split[-2]) > boundary:
					valid_fnames.append("%s/%s" % (self.path, ifname))

			if len(valid_fnames) > 0:
				input_for_cmd = ",".join(valid_fnames)
				# execute the cmd in a different process
				cmd = []
				cmd += ["gstlal_cohfar_calc_fap"]
				cmd += ["--input", input_for_cmd]
				cmd += ["--input-format", "stats"]
				cmd += ["--output", self.output[i]]
				cmd += ["--ifos", self.ifos]
				cmd += ["--duration", str(collection_time)]
				print cmd
				proc = subprocess.Popen(cmd)
				self.procs.append(proc)

	def combine_stats(self, cur_buftime):
		# list all the bank stats files in the path
		if not os.path.isfile("./combine_stats.sh"):
			print "./combine_stats.sh not found, exiting"
			sys.exit()

		cmd = []
		cmd += ["./combine_stats.sh"]
		cmd += [self.path]
		print cmd
		proc = subprocess.Popen(cmd)
		self.procs.append(proc)


class FinalSink(object):
	def __init__(self, pipeline, need_online_perform, ifos, path, output_prefix, far_factor, cluster_window = 0.5, snapshot_interval = None, background_update_interval = None, cohfar_accumbackground_output_prefix = None, cohfar_assignfar_input_fname = "marginalized_stats", background_collection_time_string = "604800,86400,7200", gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal_spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", verbose = False):
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
		# FIXME: hard-coded chisq_ratio_thresh to veto 
		self.chisq_ratio_thresh = 100
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
		self.t_bsupdate_start = None
		self.bsupdate_interval = background_update_interval
		self.bsupdater = BackgroundStatsUpdater(path = path, input_prefix_list = cohfar_accumbackground_output_prefix, output_list_string = cohfar_assignfar_input_fname, collection_time_string = background_collection_time_string, ifos = ifos)

		# online information performer
		self.need_online_perform = need_online_perform
		self.onperformer = OnlinePerformer(parent_lock = self.lock)

		# trigger control
		self.trigger_control_doc = "trigger_control.txt"
		self.last_trigger = []
		self.last_submitted_trigger = []
		self.last_trigger.append((0, 1))
		self.last_submitted_trigger.append((0, 1))

	def appsink_new_buffer(self, elem):
		with self.lock:
			buf = elem.emit("pull-buffer")
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			newevents = postcohinspiraltable.PostcohInspiralTable.from_buffer(buf)
			self.need_candidate_check = False

			nevent = len(newevents)
			#print >> sys.stderr, "%f nevent %d" % (buf_timestamp, nevent)
			# initialization
			if self.is_first_buf:
				self.t_snapshot_start = buf_timestamp
				self.t_bsupdate_start = buf_timestamp
				self.t_start = buf_timestamp
				self.is_first_buf = False

			if self.is_first_event and nevent > 0:
				self.boundary = buf_timestamp + self.cluster_window
				self.is_first_event = False

			# extend newevents to cur_event_table
			self.cur_event_table.extend(newevents)

			if self.cluster_window == 0:
				self.postcoh_table.extend(newevents)
				del self.cur_event_table[:]

			# the logic of clustering here is quite complicated, fresh up
			# yourself before reading the code
			# check if the newevents is over boundary
			while self.cluster_window > 0 and self.boundary and buf_timestamp > self.boundary:
				self.cluster(self.cluster_window)

				if self.need_candidate_check:
					self.nevent_clustered += 1
					self.__set_far(self.candidate)
					self.postcoh_table.append(self.candidate)	
					# FIXME: Currently hard-coded for single detector far H and L
					if self.gracedb_far_threshold and self.candidate.far > 0 and self.candidate.far < self.gracedb_far_threshold and self.candidate.far_h < 1E-2 and self.candidate.far_l < 1E-2 and self.__chisq_ratio_veto(self.candidate) is False:
						self.__do_gracedb_alert(self.candidate)
					if self.need_online_perform:
						self.onperformer.update_eye_candy(self.candidate)
					self.candidate = None
					self.need_candidate_check = False

			# dump zerolag candidates when interval is reached
			self.snapshot_duration = buf_timestamp - self.t_snapshot_start
			if self.snapshot_interval is not None and self.snapshot_duration >= self.snapshot_interval:
				snapshot_filename = self.get_output_filename(self.output_prefix, self.t_snapshot_start, self.snapshot_duration)
				self.snapshot_output_file(snapshot_filename)
				self.t_snapshot_start = buf_timestamp
				self.bsupdater.combine_stats(buf_timestamp)
				self.nevent_clustered = 0

			# do bsupdate when interval is reached
			bsupdate_duration = buf_timestamp - self.t_bsupdate_start
			if self.bsupdate_interval is not None and bsupdate_duration >= self.bsupdate_interval:
				self.bsupdater.update_fap_stats(buf_timestamp)
				self.t_bsupdate_start = buf_timestamp

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
		candidate.far = (max(candidate.far_2h, candidate.far_1d, candidate.far_1w)) * self.far_factor

	def __chisq_ratio_veto(self, candidate):
		chisq_ratio = candidate.chisq_H/candidate.chisq_L
		if chisq_ratio > self.chisq_ratio_thresh or chisq_ratio < 1.0/self.chisq_ratio_thresh:
			return True
		else:
			return False

	def __need_trigger_control(self, trigger):
		# do trigger control
		# FIXME: implement a sql solution for node communication ?

		with open(self.trigger_control_doc, "r") as f:
		  content = f.read().splitlines()
		
		is_submitted_idx = -1
		if len(content) > 0:
		  (last_time, last_far, is_submitted) = content[-1].split(",")
		  last_time = float(last_time)
		  last_far = float(last_far)
		  while is_submitted == "0" and len(content) + is_submitted_idx > 0:
		    is_submitted_idx = is_submitted_idx - 1;
		    (last_time, last_far, is_submitted) = content[is_submitted_idx].split(",")
		  last_time = float(last_time)
		  last_far = float(last_far)
		else:
		  last_time = self.last_trigger[-1][0]
		  last_far = self.last_trigger[-1][1]

		last_submitted_time = last_time
		last_submitted_far = last_far

		trigger_is_submitted = 0

		# suppress the trigger 
		# if it is not one order of magnitude more significant than the last trigger 
		# or if it not more significant the last submitted trigger
		if ((abs(float(trigger.end) - last_time) < 50 and abs(trigger.far/last_far) > 0.5)) or (abs(float(trigger.end) - float(last_submitted_time)) < 3600 and trigger.far > last_submitted_far*0.5) :
			print >> sys.stderr, "trigger controled, time %f, FAR %f, last_far %f, last_submitted time %f, last_submitted far %f" % (float(trigger.end), trigger.far, last_far, last_submitted_time, last_submitted_far)
			self.last_trigger.append((trigger.end, trigger.far))
			line = "%f,%e,%d\n" % (float(trigger.end), trigger.far, trigger_is_submitted)
			with open(self.trigger_control_doc, "a") as f:
			  f.write(line)
			return True
		
		print >> sys.stderr, "trigger passed, time %f, FAR %f, last_far %f, last_submitted time %f, last_submitted_far %f" % (float(trigger.end), trigger.far, last_far, last_submitted_time, last_submitted_far)

		trigger_is_submitted = 1
		#self.last_trigger.append((trigger.end, trigger.far))
		#self.last_submitted_trigger.append((trigger.end, trigger.far))
		line = "%f,%e,%d\n" % (float(trigger.end), trigger.far, trigger_is_submitted)
		with open(self.trigger_control_doc, "a") as f:
		  f.write(line)
	
		return False


	def __do_gracedb_alert(self, trigger):

		if self.__need_trigger_control(trigger):
			return
			
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
		xmldoc.childNodes[-1].appendChild(lsctables.New(postcoh_table_def.PostcohInspiralTable))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentDefTable, columns = ligolw_segments.LigolwSegmentList.segment_def_columns))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentSumTable, columns = ligolw_segments.LigolwSegmentList.segment_sum_columns))
		#xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SegmentTable, columns = ligolw_segments.LigolwSegmentList.segment_columns))

		sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(xmldoc)
		coinc_def_table = lsctables.CoincDefTable.get_table(xmldoc)
		coinc_table = lsctables.CoincTable.get_table(xmldoc)
		coinc_map_table = lsctables.CoincMapTable.get_table(xmldoc)
		coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(xmldoc)
		time_slide_table = lsctables.TimeSlideTable.get_table(xmldoc)
		postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(xmldoc)


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
		row.coa_phase = trigger.coaphase_L
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
		row.chisq = trigger.chisq_L
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
		row.coa_phase = trigger.coaphase_H
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
		row.chisq = trigger.chisq_H
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
		row.instruments = ','.join(re.findall('..',trigger.ifos)) #FIXME: for more complex detector names
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
		row.ifos = ','.join(re.findall('..',trigger.ifos)) #FIXME: for more complex detector names
		coinc_inspiral_table.append(row)

		row = coinc_map_table.RowType()
		row.event_id = "sngl_inspiral:event_id:1"
		row.table_name = "sngl_inspiral"
		row.coinc_event_id = "coinc_event:coinc_event_id:1"
		coinc_map_table.append(row)

		row = coinc_map_table.RowType()
		row.event_id = "sngl_inspiral:event_id:2"
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

		postcoh_table.append(trigger)

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
		#message2 = file(filename, "w")
		#pdb.set_trace()
		ligolw_utils.write_fileobj(xmldoc, message, gz = False, trap_signals = None)
		ligolw_utils.write_filename(xmldoc, filename, gz = False, trap_signals = None)
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
		log_message = "Optimal ra and dec from this coherent pipeline: (%f, %f) in degrees" % (trigger.ra, trigger.dec)
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
		#FIXME: for more complex detector names
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
		fname = "%s_%d_%d.xml.gz" % (output_prefix, t_snapshot_start, snapshot_duration)
		return fname

	def snapshot_output_file(self, filename, verbose = False):
		self.__wait_internal_process_finish()
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

	def __wait_internal_process_finish(self):
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()
	
		if self.bsupdater.procs is not None:
			for proc in self.bsupdater.procs:
				if proc.poll() is None:
					proc.wait()

	def write_output_file(self, filename = None, verbose = False):
		self.__wait_internal_process_finish()
		self.__write_output_file(filename, verbose = verbose)

	def __write_output_file(self, filename = None, verbose = False):
		if filename is not None:
			self.postcoh_document.set_filename(filename)
		self.postcoh_document.write_output_file(verbose = verbose)
