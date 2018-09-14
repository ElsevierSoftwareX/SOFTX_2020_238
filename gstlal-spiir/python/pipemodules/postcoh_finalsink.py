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
	from ligo.gracedb.rest import GraceDb
except ImportError:
	print >>sys.stderr, "warning: gracedb import failed, program will crash if gracedb uploads are attempted"
	GraceDb = None

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
from lal import LIGOTimeGPS

from gstlal import bottle
from gstlal import reference_psd
from gstlal.pipemodules.postcohtable import postcoh_table_def 
from gstlal.pipemodules.postcohtable import postcohtable
from gstlal.pipemodules import pipe_macro

lsctables.LIGOTimeGPS = LIGOTimeGPS

#
# =============================================================================
#
#						 glue.ligolw Content Handlers
#
# =============================================================================
#


class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass
ligolw_array.use_in(LIGOLWContentHandler)
ligolw_param.use_in(LIGOLWContentHandler)
lsctables.use_in(LIGOLWContentHandler)

#
class SegmentDocument(object):
	def __init__(self, ifos, verbose = False):

		self.get_another = lambda: SegmentDocument(ifos, verbose = verbose)

		self.filename = None
		#
		# build the XML document
		#

		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())

		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, "gstlal_inspiral_postcohspiir_online", {})
		self.segtype = pipe_macro.ONLINE_SEG_TYPE_NAME
		self.seglistdict = {self.segtype: segments.segmentlistdict((instrument, segments.segmentlist()) for instrument in re.findall('..', ifos))}


	def set_filename(self, filename):
		self.filename = filename

	def write_output_file(self, verbose = False):
		assert self.filename is not None
		with ligolw_segments.LigolwSegments(self.xmldoc, self.process) as llwsegments:
			for segtype, one_type_dict in self.seglistdict.items():
				llwsegments.insert_from_segmentlistdict(one_type_dict, name = segtype, comment = "SPIIR postcoh snapshot")
		ligolw_process.set_process_end_time(self.process)
		ligolw_utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)

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


class FAPUpdater(object):
	def __init__(self, path, input_prefix_list, ifos, output_list_string = None, collect_walltime_string = None, verbose = None):
		self.path = path
		self.input_prefix_list = input_prefix_list
		self.ifos = ifos
		self.procs_combine_stats = []
		self.procs_update_fap_stats = []
		self.output = []
		if output_list_string is not None:
			self.output = output_list_string.split(",")

		self.collect_walltime = []
		self.rm_fnames = []
		if collect_walltime_string is not None:
			times = collect_walltime_string.split(",")
			for itime in times:
				self.collect_walltime.append(int(itime))

		if self.output and len(self.output) != len(self.collect_walltime):
			raise ValueError("number of input walltimes does match the number of input filenames: %s does not match %s" % (collect_walltime_string, output_list_string))

		self.verbose = verbose


	def wait_last_process_finish(self, procs):
		if len(procs) > 0:
			for proc in procs:
				if proc.poll() is None:
					proc.wait()
		
		# delete all update processes when they are finished
		del procs[:]

	def get_fnames(self, keyword):
		# both update_fap_stats and combine_stats need to access latest cleaned
		# uped stats files
		# make sure need-to-remove files have been removed
		self.wait_last_process_finish(self.procs_combine_stats)
		# remove files that have been combined from last process
		map(lambda x: os.remove(x), self.rm_fnames)
		self.rm_fnames = []

		ls_proc = subprocess.Popen(["ls", self.path], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		ls_out = ""
		try:
			ls_out = subprocess.check_output(["grep", keyword], stdin = ls_proc.stdout)
		except:
			print "no %s file yet" % keyword
			return
		ls_proc.wait()
		ls_fnames = ls_out.split("\n")
		# remove file names that contain "next" which are temporary files
		valid_fnames = [one_fname for one_fname in ls_fnames if not re.search("next", one_fname)]
		return valid_fnames

	def update_fap_stats(self, cur_buftime):
		self.wait_last_process_finish(self.procs_update_fap_stats)
		# list all the files in the path
		#nprefix = len(self.input_prefix_list[0].split("_"))
		# FIXME: hard-coded keyword, assuming name name e.g. bank16_stats_1187008882_1800.xml.gz
		ls_fnames = self.get_fnames("stats")
		if ls_fnames is None:
			return

		for (i, collect_walltime) in enumerate(self.collect_walltime):
			boundary = cur_buftime - collect_walltime
			# find the files within the collection time
			valid_fnames = []
			for ifname in ls_fnames:
				ifname_split = ifname.split("_")
				# FIXME: hard coded the stats name e.g. bank16_stats_1187008882_1800.xml.gz
				if len(ifname_split)> 1 and ifname[-4:] != "next" and ifname_split[-2].isdigit() and int(ifname_split[-2]) > boundary:
					valid_fnames.append("%s/%s" % (self.path, ifname))

			if len(valid_fnames) > 0:
				input_for_cmd = ",".join(valid_fnames)
				# execute the cmd in a different process
				proc = self.call_calcfap(self.output[i], input_for_cmd, self.ifos, collect_walltime, verbose = self.verbose)
				self.procs_update_fap_stats.append(proc)

	def call_calcfap(self, fout, fin, ifos, walltime, verbose = False):
		cmd = []
		cmd += ["gstlal_cohfar_calc_fap"]
		cmd += ["--input", fin]
		cmd += ["--input-format", "stats"]
		cmd += ["--output", fout]
		cmd += ["--ifos", ifos]
		if verbose:
			print cmd
		proc = subprocess.Popen(cmd)
		return proc

	# combine stats every day
	def combine_stats(self, combine_duration = 86400):
		# max number of files to be combined
		max_nfile_input = 6
		if self.verbose:
			print "combining %s" % self.path
		ls_fnames = self.get_fnames("bank")
		if ls_fnames is None:
			return
	
		# FIXME: decode information assuming fixed stats name e.g. bank16_stats_1187008882_1800.xml.gz
		# decode to {'16', ['bank16_stats_1187008882_1800.xml.gz', '..']}
		stats_dict = {}
		for ifname in ls_fnames:
			this_bankid = ifname.split('_')[0][4:]
			stats_dict.setdefault(this_bankid, []).append(ifname)
	
		if '' in stats_dict.keys():
			del stats_dict['']
	
		for bankid,bank_fnames in stats_dict.items():
			collected_fnames = []
			for one_bank_fname in bank_fnames: 
				this_walltime = int(one_bank_fname.split('.')[0].split('_')[-1])
				collected_walltimes = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), collected_fnames))
				total_collected_walltime = sum(collected_walltimes)
				if this_walltime >= combine_duration:
					continue
				elif len(collected_fnames) >= max_nfile_input or total_collected_walltime >= combine_duration:
					if self.verbose:
						print "combining %s" % ','.join(collected_fnames)
					start_banktime = int(collected_fnames[0].split('_')[2])
					fout = "%s/bank%s_stats_%d_%d.xml.gz" % (self.path, bankid, start_banktime, total_collected_walltime)

					proc = self.call_calcfap(fout, ','.join(collected_fnames), self.ifos, total_collected_walltime, self.verbose)
					self.procs_combine_stats.append(proc)
					# mark to remove collected_fnames
					for frm in collected_fnames:
						self.rm_fnames.append(frm)
					collected_fnames = []
					collected_fnames.append("%s/%s" % (self.path, one_bank_fname))
				else:
					collected_fnames.append("%s/%s" % (self.path, one_bank_fname))

class FinalSink(object):
	def __init__(self, channel_dict, process_params, pipeline, need_online_perform, path, output_prefix, output_name, far_factor, cluster_window = 0.5, snapshot_interval = None, fapupdater_interval = None, cohfar_accumbackground_output_prefix = None, cohfar_accumbackground_output_name = None, fapupdater_output_fname = None, fapupdater_collect_walltime_string = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal-spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", output_skymap = 0, superevent_thresh = 3.8e-7, verbose = False):
		#
		# initialize
		#
		self.lock = threading.Lock()
		self.verbose = verbose
		self.pipeline = pipeline
		self.is_first_buf = True
		self.is_first_event = True
		self.channel_dict = channel_dict
		self.ifos = lsctables.ifos_from_instrument_set(channel_dict.keys()).replace(",", "") # format: "H1L1V1"

		# cluster parameters
		self.cluster_window = cluster_window
		self.candidate = None
		self.cluster_boundary = None
		self.need_candidate_check = False
		self.cur_event_table = lsctables.New(postcoh_table_def.PostcohInspiralTable)
		# FIXME: hard-coded chisq_ratio_thresh to veto 
		self.chisq_ratio_thresh = 100
		self.superevent_thresh = superevent_thresh
		self.nevent_clustered = 0

		# gracedb parameters
		self.far_factor = far_factor
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url
		if GraceDb:
			self.gracedb_client = GraceDb(gracedb_service_url)

		# keep a record of segments and is snapshotted
		# our segments is determined by if incoming buf is GAP
		self.seg_document = SegmentDocument(self.ifos)

		# the postcoh doc stores clustered postcoh triggers and is snapshotted
		self.postcoh_document = PostcohDocument()
		self.postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

		# deprecated: save the last 30s zerolags to help check the significance of current candidate
		# hard-coded to be 30s to be consistent with iDQ range
		# self.lookback_event_table = lsctables.New(postcoh_table_def.PostcohInspiralTable)
		# self.lookback_window = 30
		# self.lookback_boundary = None
		# coinc doc to be uploaded to gracedb
		self.coincs_document = CoincsDocFromPostcoh(path, process_params, channel_dict)
		# snapshot parameters
		self.path = path
		self.output_prefix = output_prefix
		self.output_name = output_name
		self.snapshot_interval = snapshot_interval
		self.thread_snapshot = None
		self.thread_snapshot_segment = None
		self.t_snapshot_start = None
		self.snapshot_duration = None

		# background updater
		self.total_duration = None
		self.t_start = None
		self.t_fapupdater_start = None
		self.fapupdater_interval = fapupdater_interval
		self.fapupdater = FAPUpdater(path = path, input_prefix_list = cohfar_accumbackground_output_prefix, output_list_string = fapupdater_output_fname, collect_walltime_string = fapupdater_collect_walltime_string, ifos = self.ifos, verbose = self.verbose)

		# online information performer
		self.need_online_perform = need_online_perform
		self.onperformer = OnlinePerformer(parent_lock = self.lock)

		# trigger control
		self.trigger_control_doc = "trigger_control.txt"
		if not os.path.exists(self.trigger_control_doc):
			file(self.trigger_control_doc, 'w').close()
		self.last_trigger = []
		self.last_submitted_trigger = []
		self.last_trigger.append((0, 1))
		self.last_submitted_trigger.append((0, 1))

		# skymap
		self.output_skymap = output_skymap

	def __pass_test(self, candidate):
		if self.candidate.far <= 0.0:
			return False

		# just submit it if is a low-significance trigger
		if self.candidate.far < self.gracedb_far_threshold and self.candidate.far > self.superevent_thresh:
			return True

		# FIXME: any two of the sngl fars need to be < 1e-2
		# single far veto for high-significance trigger
		ifo_active=[self.candidate.snglsnr_H!=0,self.candidate.snglsnr_L!=0,self.candidate.snglsnr_V!=0]
		ifo_fars_ok=[self.candidate.far_h < 1E-2, self.candidate.far_l < 1E-2, self.candidate.far_v < 1E-2]
		ifo_chisqs=[self.candidate.chisq_H,self.candidate.chisq_L,self.candidate.chisq_V]
		if self.candidate.far < self.superevent_thresh:
			return sum([i for (i,v) in zip(ifo_fars_ok,ifo_active) if v])>=2 and all((lambda x: [i1/i2 < self.chisq_ratio_thresh for i1 in x for i2 in x])([i for (i,v) in zip(ifo_chisqs,ifo_active) if v]))


	def appsink_new_buffer(self, elem):
		with self.lock:
			buf = elem.emit("pull-buffer")
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			newevents = postcohtable.GSTLALPostcohInspiral.from_buffer(buf)
			self.need_candidate_check = False

			# NOTE: the first entry is used to add to the segments, not a really event
			participating_ifos = re.findall('..', newevents[0].ifos)
			buf_seg = segments.segment(buf_timestamp, buf_timestamp + LIGOTimeGPS(0, buf.duration))
			for segtype, one_type_dict in self.seg_document.seglistdict.items():
				for ifo in one_type_dict.keys():
					if ifo in participating_ifos:
						this_seglist = one_type_dict[ifo]
						this_seglist = this_seglist + segments.segmentlist([buf_seg])
						this_seglist.coalesce()
						one_type_dict[ifo] = this_seglist

			# remove the first event entry
			newevents = newevents[1:]
			nevent = len(newevents)

			# print >> sys.stderr, "%f nevent %d" % (buf_timestamp, nevent)
			# initialization
			if self.is_first_buf:
				self.t_snapshot_start = buf_timestamp
				self.t_fapupdater_start = buf_timestamp
				self.t_start = buf_timestamp
				self.is_first_buf = False

			if self.is_first_event and nevent > 0:
				self.cluster_boundary = buf_timestamp + self.cluster_window
				self.is_first_event = False

			# extend newevents to cur_event_table and event_table_30s
			self.cur_event_table.extend(newevents)
			# self.lookback_boundary = buf_timestamp - self.lookback_window
			# self.lookback_event_table.extend(newevents)
			# iterutils.inplace_filter(lambda row: row.end > self.lookback_boundary, self.lookback_event_table)

			if self.cluster_window == 0:
				self.postcoh_table.extend(newevents)
				del self.cur_event_table[:]

			# NOTE: only consider clustered trigger for uploading to gracedb 
			# check if the newevents is over boundary
			# this loop will exit when the cluster_boundary is incremented to be > the buf_timestamp, see plot in self.cluster()

			while self.cluster_window > 0 and self.cluster_boundary and buf_timestamp > self.cluster_boundary:
				self.cluster(self.cluster_window)

				if self.need_candidate_check:
					self.nevent_clustered += 1
					self.__set_far(self.candidate)
					self.postcoh_table.append(self.candidate)
					if self.gracedb_far_threshold and self.__pass_test(self.candidate):
						self.__do_gracedb_alert(self.candidate)
					if self.need_online_perform:
						self.onperformer.update_eye_candy(self.candidate)
					self.candidate = None
					self.need_candidate_check = False

			# dump zerolag candidates when interval is reached
			self.snapshot_duration = buf_timestamp - self.t_snapshot_start
			if self.snapshot_interval is not None and self.snapshot_duration >= self.snapshot_interval:
				snapshot_filename = self.get_output_filename(self.output_prefix, self.output_name, self.t_snapshot_start, self.snapshot_duration)
				self.snapshot_output_file(snapshot_filename)
				self.snapshot_segment_file(self.t_snapshot_start, self.snapshot_duration)
				self.t_snapshot_start = buf_timestamp
				self.nevent_clustered = 0
				# also combine background_stats files so we don't end up with too many files
				self.fapupdater.combine_stats()

			# do calcfap when interval is reached
			fapupdater_duration = buf_timestamp - self.t_fapupdater_start
			if self.fapupdater_interval is not None and fapupdater_duration >= self.fapupdater_interval:
				self.fapupdater.update_fap_stats(buf_timestamp)
				self.t_fapupdater_start = buf_timestamp


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
		# send candidate to be gracedb checked only when:
		# time ->->->->
		#                     |buf_timestamp
		#          ___________(cur_table)
		#                |boundary
		#           |candidate to be gracedb checked = peak of cur_table < boundary
		#                  |candidate remain = peak of cur_table > boundary
		# afterwards:
		#                     |buf_timestamp
		#                 ____(cur_table cleaned)
		#                           |boundary incremented

		# always choose the head event to test its end against boundary
		if self.candidate is None:
			self.candidate = self.__select_head_event()

		# make sure the candidate is within the boundary
		if self.candidate is None or self.candidate.end > self.cluster_boundary:
			self.cluster_boundary = self.cluster_boundary + cluster_window
			self.candidate = None # so we can reselect a candidate next time
			return
		# the first event in cur_event_table
		peak_event = self.__select_head_event()
		# find the max cohsnr event within the boundary of cur_event_table
		for row in self.cur_event_table:
			if row.end <= self.cluster_boundary and row.cohsnr > peak_event.cohsnr:
				peak_event = row

		# cur_table is empty and we do have a candidate, so need to check the candidate
		if peak_event is None:
			# no event within the boundary, candidate is the peak, update boundary
			self.cluster_boundary = self.cluster_boundary + cluster_window
			self.need_candidate_check = True
			return

		if peak_event.end <= self.cluster_boundary and peak_event.cohsnr > self.candidate.cohsnr:
			self.candidate = peak_event
			iterutils.inplace_filter(lambda row: row.end > self.cluster_boundary, self.cur_event_table)
			# update boundary
			self.cluster_boundary = self.candidate.end + cluster_window
			self.need_candidate_check = False
		else:
			iterutils.inplace_filter(lambda row: row.end > self.cluster_boundary, self.cur_event_table)
			# update boundary
			self.cluster_boundary = self.cluster_boundary + cluster_window
			self.need_candidate_check = True

	def __set_far(self, candidate):
		candidate.far = (max(candidate.far_2h, candidate.far_1d, candidate.far_1w)) * self.far_factor
		candidate.far_h = (max(candidate.far_h_2h, candidate.far_h_1d, candidate.far_h_1w)) * self.far_factor
		candidate.far_l = (max(candidate.far_l_2h, candidate.far_l_1d, candidate.far_l_1w)) * self.far_factor
		candidate.far_v = (max(candidate.far_v_2h, candidate.far_v_1d, candidate.far_v_1w)) * self.far_factor

	# def __lookback_far(self, candidate):
		# FIXME: hard-code to check event that's < 5e-7
		# if candidate.far > 5e-7:
		#	 return
		# else:
		#	 count_events = sum((lookback_event.far < 1e-4) for lookback_event in self.lookback_event_table)
		#	 if count_events > 1:
		#		 # FAR estimation is not valide for this period, increase the FAR
		#		 # FIXME: should derive FAR from count_events
		#		  candidate.far = 9.99e-6

			# all_snr_H = self.lookback_event_table.getColumnByName('snglsnr_H')
			# all_snr_L = self.lookback_event_table.getColumnByName('snglsnr_L')
			# all_snr_V = self.lookback_event_table.getColumnByName('snglsnr_V')
			# all_chisq_H = self.lookback_event_table.getColumnByName('chisq_H')
			# all_chisq_L = self.lookback_event_table.getColumnByName('chisq_L')
			# all_chisq_V = self.lookback_event_table.getColumnByName('chisq_V')
			# count_better_H = sum((snr > candidate.snglsnr_H && chisq < candidate.chisq_H) for (snr, chisq) in zip(all_snr_H, allchisq_H)) 
			# count_better_L = sum((snr > candidate.snglsnr_L && chisq < candidate.chisq_L) for (snr, chisq) in zip(all_snr_L, allchisq_L)) 
			# count_better_V = sum((snr > candidate.snglsnr_V && chisq < candidate.chisq_V) for (snr, chisq) in zip(all_snr_V, allchisq_V)) 
			# if count_better_H > 0 or count_better_L > 0 or count_better_V > 0:
			#	 candidate.far = 9.99e-6
	
	
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
		gracedb_ids = []
		common_messages = []

		self.coincs_document.assemble_tables(trigger)
		xmldoc = self.coincs_document.xmldoc
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
		ligolw_utils.write_fileobj(xmldoc, message, gz = False)
		ligolw_utils.write_filename(xmldoc, filename, gz = False, trap_signals = None)
		xmldoc.unlink()
	
		print >>sys.stderr, "sending %s to gracedb ..." % filename
		gracedb_upload_itrial = 1
		# FIXME: make this optional from cmd line?
		while gracedb_upload_itrial < 10:
			try:
				resp = self.gracedb_client.createEvent(self.gracedb_group, self.gracedb_pipeline, filename, filecontents = message.getvalue(), search = self.gracedb_search)
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

		# delete the xmldoc and get a new empty one for next upload
		coincs_document = self.coincs_document.get_another()
		del self.coincs_document
		self.coincs_document = coincs_document
		if not gracedb_ids:
			print "gracedb upload of %s failed completely" % filename
			return
		gracedb_id = gracedb_ids[0]
		log_message = "Optimal ra and dec from this coherent pipeline: (%f, %f) in degrees" % (trigger.ra, trigger.dec)
		while gracedb_upload_itrial < 10:
			try:
				resp = self.gracedb_client.writeLog(gracedb_id, log_message , filename = None, tagname = "analyst_comments")
				if resp.status != httplib.CREATED:
						print >>sys.stderr, "gracedb upload of log failed"
				else:
					break
			except:
				gracedb_upload_itrial += 1

		# FIXME: upload skymap if output_skymap is turned on
#		if self.output_skymap == 1:
			#skymap_loc = "%s/%s_%d_%d_%d" % (skymap_url, trigger.pivotal_ifo, trigger.end_time, trigger.end_time_ns, trigger.tmplt_idx)


		if self.verbose:
			print >>sys.stderr, "retrieving PSDs from whiteners and generating psd.xml.gz ..."
		psddict = {}
		#FIXME: for more complex detector names
		instruments = re.findall('..', trigger.ifos)
		for instrument in instruments:
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


		#
		# do PSD and ranking data file uploads
		#

		while common_messages:
			message, filename, tag, contents = common_messages.pop()
			gracedb_upload_itrial = 1
			gracedb_id = gracedb_ids[0]
			while gracedb_upload_itrial < 10:
				try:
					resp = self.gracedb_client.writeLog(gracedb_id, message, filename = filename, filecontents = contents, tagname = tag)
					resp_json = resp.json()
					if resp.status != httplib.CREATED:
						print >>sys.stderr, "gracedb upload of %s for ID %s failed" % (filename, gracedb_id)
					else:
						break
				except:
					gracedb_upload_itrial += 1

	def get_output_filename(self, output_prefix, output_name, t_snapshot_start, snapshot_duration):
		if output_prefix is not None:
			fname = "%s_%d_%d.xml.gz" % (output_prefix, t_snapshot_start, snapshot_duration)
			return fname
		assert output_name is not None
		return output_name

	def snapshot_segment_file(self, t_snapshot_start, duration, verbose = False):
		filename = "%s/%s_SEGMENTS_%d_%d.xml.gz" % (self.path, self.ifos, t_snapshot_start, duration)
		# make sure the last round of output dumping is finished 
		if self.thread_snapshot_segment is not None and self.thread_snapshot_segment.isAlive():
			self.thread_snapshot_segment.join()
	
		# free the last used memory
		del self.thread_snapshot_segment
		# copy the memory
		seg_document_cpy = self.seg_document
		seg_document_cpy.set_filename(filename)
		# free thread context
		# start new thread
		self.thread_snapshot_segment = threading.Thread(target = seg_document_cpy.write_output_file, args =(seg_document_cpy, ))
		self.thread_snapshot_segment.start()

		# set a new document for seg_document
		seg_document = self.seg_document.get_another()
		# remember to delete the old seg doc
		del self.seg_document
		self.seg_document = seg_document

	def snapshot_output_file(self, filename, verbose = False):
		# make sure the last round of output dumping is finished 
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()
	
		# copy the memory
		postcoh_document_cpy = self.postcoh_document
		postcoh_document_cpy.set_filename(filename)
		# free thread context
		del self.thread_snapshot
		self.thread_snapshot = threading.Thread(target = postcoh_document_cpy.write_output_file, args =(postcoh_document_cpy, ))
		self.thread_snapshot.start()

		# set a new document for postcoh_document
		postcoh_document = self.postcoh_document.get_another()
		# remember to delete the old postcoh doc
		del self.postcoh_table
		del self.postcoh_document
		self.postcoh_document = postcoh_document
		self.postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

	def __wait_internal_process_finish(self):
		if self.thread_snapshot is not None and self.thread_snapshot.isAlive():
			self.thread_snapshot.join()

		if self.thread_snapshot_segment is not None and self.thread_snapshot_segment.isAlive():
			self.thread_snapshot_segment.join()
	
		self.fapupdater.wait_last_process_finish(self.fapupdater.procs_update_fap_stats)
		self.fapupdater.wait_last_process_finish(self.fapupdater.procs_combine_stats)

	def write_output_file(self, filename = None, verbose = False):
		self.__wait_internal_process_finish()
		self.__write_output_file(filename, verbose = verbose)

	def __write_output_file(self, filename = None, verbose = False):
		if filename is not None:
			self.postcoh_document.set_filename(filename)
		self.postcoh_document.write_output_file(verbose = verbose)
		# FIXME: hard-coded segment filename
		seg_filename = "%s/%s_SEGMENTS_%d_%d.xml.gz" % (self.path, self.ifos, self.t_snapshot_start, self.snapshot_duration)
		self.seg_document.set_filename(seg_filename)
		self.seg_document.write_output_file(verbose = verbose)

class CoincsDocFromPostcoh(object):
	sngl_inspiral_columns = ("process_id", "ifo", "end_time", "end_time_ns", "eff_distance", "coa_phase", "mass1", "mass2", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "sigmasq", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id", "Gamma0", "Gamma1")

	def __init__(self, url, process_params, channel_dict, comment = None, verbose = False):
		#
		# build the XML document
		#
		self.get_another = lambda: CoincsDocFromPostcoh(url = url, process_params = process_params, channel_dict = channel_dict, comment = comment, verbose = verbose)
	
		self.channel_dict = channel_dict
		self.url = url
		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, u"gstlal_inspiral_postcohspiir_online", process_params, comment = comment, ifos = channel_dict.keys())
	
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = self.sngl_inspiral_columns))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(postcoh_table_def.PostcohInspiralTable))

	# path here is the job id
	def assemble_tables(self, trigger):
		self.assemble_snglinspiral_table(trigger)
		coinc_def_table = lsctables.CoincDefTable.get_table(self.xmldoc)
		coinc_table = lsctables.CoincTable.get_table(self.xmldoc)
		coinc_inspiral_table = lsctables.CoincInspiralTable.get_table(self.xmldoc)
		postcoh_table = postcoh_table_def.PostcohInspiralTable.get_table(self.xmldoc)

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
		row.process_id = self.process.process_id
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

		self.assemble_coinc_map_table(trigger)
		self.assemble_time_slide_table(trigger)

		postcoh_table.append(trigger)

	def assemble_coinc_map_table(self, trigger):

		coinc_map_table = lsctables.CoincMapTable.get_table(self.xmldoc)
		iifo = 0
		# FIXME: hard-coded ifo length
		for ifo in re.findall('..', trigger.ifos):
			row = coinc_map_table.RowType()
			row.event_id = "sngl_inspiral:event_id:%d" % iifo
			row.table_name = "sngl_inspiral"
			row.coinc_event_id = "coinc_event:coinc_event_id:1"
			coinc_map_table.append(row)
			iifo += 1

	def assemble_time_slide_table(self, trigger):

		time_slide_table = lsctables.TimeSlideTable.get_table(self.xmldoc)
		# FIXME: hard-coded ifo length
		for ifo in re.findall('..', trigger.ifos):
			row = time_slide_table.RowType()
			row.instrument = ifo
			row.time_slide_id = "time_slide:time_slide_id:6"
			row.process_id = self.process.process_id
			row.offset = 0
			time_slide_table.append(row)


	def assemble_snglinspiral_table(self, trigger):
		sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(self.xmldoc)
		for standard_column in ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "event_id"):
			try:
				sngl_inspiral_table.appendColumn(standard_column)
			except ValueError:
				# already has it
				pass

		# FIXME: hard-coded ifo len == 2
		iifo = 0
		for ifo in re.findall('..', trigger.ifos):
			row = sngl_inspiral_table.RowType()
			# Setting the individual row
			row.process_id = self.process.process_id
			row.ifo =  ifo
			row.search = self.url
			row.channel = self.channel_dict[ifo]
			row.end_time = getattr(trigger, "end_time_%s" % ifo[0])
			row.end_time_ns = getattr(trigger, "end_time_ns_%s" % ifo[0])
			row.end_time_gmst = 0 
			row.impulse_time = 0
			row.impulse_time_ns = 0
			row.template_duration = trigger.template_duration
			row.event_duration = 0
			row.amplitude = 0
			row.eff_distance = getattr(trigger, "deff_%s" % ifo[0])
			row.coa_phase = getattr(trigger, "coaphase_%s" % ifo[0])
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
			row.snr = getattr(trigger, "snglsnr_%s" % ifo[0])
			row.chisq = getattr(trigger, "chisq_%s" % ifo[0])
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
			row.event_id = "sngl_inspiral:event_id:%d" % iifo
			sngl_inspiral_table.append(row)
			iifo +=  1
