# Copyright (C) 2014 Chris Pankow
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
"""Handler and parts for stream-based excess power searches"""

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import os
import sys
import time
import tempfile
import threading
import math
import json
import types

from optparse import OptionParser, OptionGroup

import ConfigParser
from ConfigParser import SafeConfigParser

import numpy

from glue import datafind

from glue.ligolw import ligolw, lsctables
lsctables.use_in(ligolw.LIGOLWContentHandler)
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import search_summary as ligolw_search_summary

from glue.segments import segment, segmentlist, segmentlistdict, PosInfinity
from glue.lal import LIGOTimeGPS, Cache, CacheEntry

from pylal import snglcluster
from pylal import ligolw_bucluster

from pylal import datatypes as laltypes
from pylal.xlal.datatypes.snglburst import SnglBurst
from lalburst import lnOneMinusChisqCdf

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst

from gstlal import pipeparts
from gstlal.simplehandler import Handler
from gstlal.reference_psd import write_psd
import utils, filters

#
# =============================================================================
#
#                                Handler Class
#
# =============================================================================L
#

class EPHandler(Handler):
	"""
	Handler class for the excess power pipeline. Keeps various bits of information that the pipeline emits and consumes. This is also in charge of signalling the rebuild of various matrices and vectors needed by the pipeline.
	"""
	def __init__(self, mainloop, pipeline):
		# FIXME: Synch with base Handler class, some of the init and message
		# handling is done better there

		# Instrument and channel
		self.inst = None
		self.channel = None

		# Book keeping
		self.start = 0
		self.stop = -1
		self.seglist = segmentlistdict()
		self.seglist["state"] = segmentlist([])
		self.current_segment = None
		# How long of a time to ignore output from the whitener stabilizing
		self.whitener_offset = 0

		# Keep track of units -- enable us to go below rates of 1 Hz
		self.units = utils.EXCESSPOWER_UNIT_SCALE['Hz']

		# Defaults -- Time-frequency map settings
		self.base_band = 16
		self.flow = 64 
		self.fhigh = 2048
		self.fft_length = 8 # s
		self.frequency_overlap = 0 # %

		# Defaults -- Resolution settings
		self.rate = 2048
		self.max_level = 1
		self.max_bandwidth = 512
		self.max_dof = None
		self.fix_dof = None

		# Defaults -- filtering
		self.filter_xml = {}
		self.filter_bank = None
		self.freq_filter_bank = None
		# TODO: Maybe not necessary
		self.firbank = None

		# Defaults -- PSD settings
		self.whitener = None
		# Current measured PSD from the whitener
		self.prev_psd = self.psd = None
		self.psd_mode = 0  # default is to adapt
		# This is used to store the previous value of the PSD power
		self.psd_power = 0
		self.cache_psd = None
		self.cache_psd_dir = "./"
		self.last_psd_cache = 0
		self.psd_change_thresh = 0.06
		# This is used to store the change from the previous PSD power
		self.psd_change = 0.0

		# Defaults -- mixer matrices
		self.chan_matrix = {}
		self.mmixers = {}

		# Defaults -- data products
		self.output = True
		self.triggers = None
		self.process_params = None
		self.process = None
		self.outdir = os.getcwd()
		self.outdirfmt = ""
		self.triggers = EPHandler.make_output_table()
		self.output_cache = Cache()
		self.output_cache_name = None
		self.snr_thresh = None
		self.event_number = 0
		self.fap = None
		self.dump_frequency = 600 # s
		self.max_events = 1e6
		self.time_since_dump = self.start

		self.trigger_segment = None

		self.spec_corr = self.build_default_correlation( self.rate )

		# required for file locking
		self.lock = threading.Lock()

		self.tempdir = None

		self.bus = pipeline.get_bus()
		self.bus.add_signal_watch()
		self.bus.connect( "message", self.process_message )

		self.pipeline = pipeline

		self.verbose = False
		self._clustering = False
		self.channel_monitoring = False
		self.stats = utils.SBStats()

		super(type(self), self).__init__(mainloop, pipeline)

	def initialize_handler_objects(self, options):
		# This is invoked here, or else the default rate is used, which will 
		# cause funny behavior for the defaults with some cases

		# Set process params in handler for use with the output xmldocs
		self.make_process_tables(options, None)

		# Set all the relevant time bookkeeping
		self.time_since_dump = self.stop = self.start

		# FIXME: This probably isn't even needed with the drop samples
		df = 1.0 / self.fft_length
		if self.psd is None:
			self.psd = EPHandler.build_default_psd(self.rate, df, self.fhigh)
		self.rebuild_filter()
		self.rebuild_matrix_mixers()

	def set_trigger_time_and_action(self, trig_seg, action="psd"):
		"""
		Inform the handler of a specific time of interest, along with a action to take.
		"""

		self.trigger_segment = segment(map(LIGOTimeGPS, trig_seg))
		# TODO: Bounds checking
		if action == "psd":
			pass
		elif action == "scan":
			pass

	def handle_segment(self, elem, timestamp, segment_type):
		"""
		Process state changes from the state vector mechanism.
		"""
		if segment_type == "on":
			self.current_segment = segment(LIGOTimeGPS(timestamp) / 1e9, PosInfinity)
			if self.verbose:
				print >>sys.stderr, "Starting segment #%d: %.9f" % (len(self.seglist["state"]), self.current_segment[0])
		elif segment_type == "off":
			if self.current_segment is None: 
				print >>sys.stderr, "Got a message to end a segment, but no current segment exists. Ignoring."
				return
			self.seglist["state"].append(
				segment(self.current_segment[0], LIGOTimeGPS(timestamp / 1e9))
			)
			# Make it very clear we don't have a segment currently
			self.current_segment = None
			if self.verbose:
				print >>sys.stderr, "Ending segment #%d: %s" % (len(self.seglist["state"])-1, str(self.seglist["state"][-1]))
		else:
			print >>sys.stderr, "Unrecognized state change, ignoring."

	def process_message(self, bus, message):
		"""
		Process a message from the bus. Depending on what it is, we may drop various data products on to disk, or perform various other actions.
		"""

		if message.type == gst.MESSAGE_EOS:
			self.shutdown(None, None)
			return
		elif message.type == gst.MESSAGE_LATENCY:
			if self.verbose:
				print >>sys.stderr, "Got latency message, ignoring for now."
			return
		elif message.type == gst.MESSAGE_TAG:
			if self.psd_mode == 1 and self.psd is not None and not self.whitener.get_property("mean-psd"):
				if self.verbose:
					print >>sys.stderr, "Got tags message, fixing current PSD to whitener."
				self.whitener.set_property("mean-psd", self.psd.data)
				self.whitener.set_property("psd-mode", self.psd_mode) # GSTLAL_PSDMODE_FIXED
			elif self.psd_mode == 1 and self.psd is not None:
				if self.verbose:
					print >>sys.stderr, "Got tags message, but already fxed current PSD to whitener."
			else:
				if self.verbose:
					print >>sys.stderr, "Got tags message, but no fixed spectrum to set, so ignoring for now."
			return
		elif message.structure is None: 
			print >>sys.stderr, "Got message with type: %s ...but no handling logic, so ignored." % str(message.type)
			return

		# TODO: Move this to PSD difference checker
		if message.structure.get_name() == "spectrum":
			# FIXME: Units
			ts = message.timestamp*1e-9
			if self.trigger_segment is not None and ts in self.trigger_segment:
				self.dump_psd(ts, self.cache_psd_dir)
			elif self.cache_psd is not None and self.cache_psd + self.last_psd_cache < ts:
				self.dump_psd(ts, self.cache_psd_dir)
				self.last_psd_cache = ts

	def dump_psd(self, timestamp, psddir="./"):
		"""
		Dump the currently cached PSD to a LIGOLW XML file. If the psddir isn't specified, it defaults to the current directory.
		"""

		filename = utils.make_cache_parseable_name(
			inst = self.inst,
			tag = self.channel + "_PSD",
			start = round(timestamp) - 40, # FIXME: calculate PSD history length
			stop = round(timestamp),
			ext = "xml.gz",
			dir = psddir
		)

		write_psd(filename, {self.inst: self.cur_psd})

	def add_firbank(self, firbank):
		"""
		Set the main base band FIR bank, and build the FIR matrix.
		"""
		firbank.set_property("fir-matrix", self.rebuild_filter())
		# Impose a latency since we've advanced the filter in the 
		# generation step. See build_filter in the excesspower library
		firbank.set_property("latency", self.filter_bank.shape[1]/2)
		self.firbank = firbank

	def add_matmixer(self, mm, res_level):
		self.mmixers[res_level] = mm
		self.rebuild_matrix_mixers(res_level)

	@staticmethod
	def build_default_psd(rate, df, fhigh):
		"""
		Builds a dummy PSD to use until we get the right one.
		"""
		psd = laltypes.REAL8FrequencySeries()
		psd.deltaF = df
		psd.sampleUnits = laltypes.LALUnit( "s strain^2" )
		psd.data = numpy.ones( int(rate/2/df) + 1 ) 
		psd.f0 = 0
		return psd

	@staticmethod
	def build_default_correlation(rate):
		"""
		Builds a Kronecker delta correlation series for k, k'.
		"""
		# FIXME: rate isn't right
		corr = numpy.zeros(rate + 1)
		corr[0] = 1
		return corr

	def rebuild_matrix_mixers(self, res_level = None):
		"""
		Rebuilds the matrix mixer matrices from the coefficients calculated in rebuild_chan_mix_matrix and assigns them to their proper element.
		"""
		for i, mm in self.mmixers.iteritems():
			if res_level is not None and res_level != i: 
				continue

			nchannels = self.filter_bank.shape[0]
			self.chan_matrix[i] = filters.build_wide_filter_norm( 
				corr = self.spec_corr, 
				freq_filters = self.freq_filter_bank,
				level = i,
				frequency_overlap = self.frequency_overlap,
				band = self.base_band
			)
			cmatrix = filters.build_chan_matrix( 
				nchannels = nchannels,
				frequency_overlap = self.frequency_overlap,
				up_factor = i,
				norm = self.chan_matrix[i]
			) 
			# DEBUG: Uncommet to get matrix mixer elements
			#cmatrix.tofile(open("matrix_level_%d" % i, "w"))
			mm.set_property("matrix", cmatrix)

	def rebuild_filter(self):
		"""
		Calling this function rebuilds the filter FIR banks and assigns them to their proper element. This is normally called when the PSD or spectrum correlation changes.
		"""
		if not self.filter_xml.has_key((0,1)):
			self.build_filter_xml(res_level = 0, ndof = 1)

		self.filter_bank, self.freq_filter_bank = filters.build_filter_from_xml(
			self.filter_xml[(0,1)],
			self.psd,
			self.spec_corr
		)
		return self.filter_bank

	def build_filter_xml(self, res_level, ndof=1, loc=None, verbose=False):
		"""
		Calls the EP library to create a XML of sngl_burst tables representing the filter banks. At the moment, this dumps them to the current directory, but this can be changed by supplying the 'loc' argument. The written filename is returned for easy use by the trigger generator.
		"""
		self.filter_xml[(res_level, ndof)] = filters.create_bank_xml(
			self.flow,
			self.fhigh,
			self.base_band*2**res_level,
			1.0 / (2*self.base_band*2**res_level), # resolution level starts from 0
			res_level,
			ndof,
			self.frequency_overlap,
			self.inst,
			1 if ndof == 1 else self.units,
		)

		if self.tempdir is None:
			self.tempdir = tempfile.mkdtemp()
		# Store the filter name so we can destroy it later
		output = os.path.join(self.tempdir, "gstlal_excesspower_bank_%s_%s_level_%d_%d.xml" % (self.inst, self.channel, res_level, ndof))
		self.filter_xml[output] = self.filter_xml[(res_level, ndof)]

		if self.mmixers.has_key(res_level):
			mmatrix = self.mmixers[res_level].get_property("matrix")
			filter_table = lsctables.SnglBurstTable.get_table(self.filter_xml[(res_level, ndof)])
			assert len(mmatrix[0]) == len(filter_table)

		# Write it
		self.lock.acquire()
		ligolw_utils.write_filename(self.filter_xml[output], 
			output, 
			verbose = verbose,
		    gz = (output or "stdout").endswith(".gz"))
		self.lock.release()

		# Just get the table we want
		self.filter_xml[(res_level, ndof)] = lsctables.SnglBurstTable.get_table(self.filter_xml[(res_level, ndof)])
		return output

	def destroy_filter_xml(self, loc=""):
		"""
		Remove the filter XML files.
		"""
		for f, v in self.filter_xml.iteritems():
			if os.path.exists(str(f)):
				os.remove(f)

	def rebuild_chan_mix_matrix(self):
		"""
		Calling this function rebuilds the matrix mixer coefficients for higher resolution components. This is normally called when the PSD or spectrum correlation changes.
		"""
		self.chan_matrix = filters.build_wide_filter_norm( 
			corr = self.spec_corr, 
			freq_filters = self.freq_filter_bank,
			level = self.max_level,
			frequency_overlap = self.frequency_overlap
		)
		return self.chan_matrix

	def rebuild_everything(self):
		"""
		Top-level function to handle the asynchronous updating of FIR banks and matrix mixer elements.
		"""
		# Rebuild filter bank and hand it off to the FIR element
		if self.verbose:
			print >>sys.stderr, "Rebuilding FIR bank"
		self.firbank.set_property("fir_matrix", self.rebuild_filter())
		latency = len(self.firbank.get_property("fir_matrix")[0])/2+1 
		self.firbank.set_property("latency", latency)
		if self.verbose:
			print >>sys.stderr, "New filter latency %d (%f s)" % (latency, latency/float(self.rate))

		if self.verbose:
			print >>sys.stderr, "Rebuilding matrix mixer"
		#self.rebuild_chan_mix_matrix()
		# Rebuild the matrix mixer with new normalization coefficients
		self.rebuild_matrix_mixers()
		if self.verbose:
			print >>sys.stderr, "...done."

	def make_process_tables(self, options=None, xmldoc=None):
		"""
		Create a process and process_params table for use in the output document. If the options parameter is passed, the process_params table will be created from it. If the output xmldoc is provided, it will register the pipeline with the document. Returns the process and process_params table, however, note that the process table will be empty if no options are provided.
		"""
		if options:
			self.process_params = vars(options)

		if self.process_params is None:
			print >>sys.stderr, "WARNING: Options have not yet been set in the handler. Process and ProcessParams may not be constructed properly. Call handler.make_process_tables() with the options argument to set command line options."
			return lsctables.New(lsctables.ProcessTable)

		if self.process is None:
			xmldoc = ligolw.Document() # dummy document
			xmldoc.appendChild(ligolw.LIGO_LW())
			self.process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_excesspower", self.process_params)
			self.process.set_ifos([self.inst])
		elif xmldoc and self.process is not None:
			# This branch is here to prevent the register_to_xmldoc method from
			# creating a new process id for each call to this function, since
			# it can be called several times during one run of the same pipeline
			# Note that this is basically the same code as register_to_xmldoc
			# it just preserves the process row
			ptable = lsctables.New(lsctables.ProcessTable)
			ptable.append( self.process )
			xmldoc.childNodes[0].appendChild(ptable)
			ligolw_process.append_process_params(xmldoc, 
				self.process, 
				ligolw_process.process_params_from_dict(self.process_params)
			)


		return self.process

	@property
	def clustering(self):
		return self._clustering

	# Ensure that if clustering is on, we add the appropriate columns to be 
	# output
	@clustering.setter
	def clustering(self, value):
		self._clustering = value
		if len(self.triggers) > 1:
			ligolw_bucluster.add_ms_columns_to_table(self.triggers)

	@classmethod
	def make_output_table(cls, add_ms_columns=False):
		tbl = lsctables.New(lsctables.SnglBurstTable, ["ifo", "peak_time", 
			"peak_time_ns", "start_time", "start_time_ns",
			"duration",  "search", "event_id", "process_id",
			"central_freq", "channel", "amplitude", "snr", "confidence",
			"chisq", "chisq_dof", "bandwidth"])
		#if add_ms_columns:
			#ligolw_bucluster.add_ms_columns_to_table(tbl)
		return tbl

	def get_triggers(self, appsink):
		"""
        Retrieve triggers from the appsink element and postprocess them, adding them to our current list.
		"""

		buf = appsink.emit("pull-buffer")

		if not self.output:
			return # We don't want event information

		# What comes out of SnglBurst.from_buffer is a
		# pylal.xlal.datatypes.snglburst.SnglBurst object. It does not have
		# all the trappings of its glue.ligolw.lsctables cousin, so we
		# convert it here first
		for event in [utils.convert_sngl_burst(sb, self.triggers) for sb in SnglBurst.from_buffer(buf)]:

			# FIXME: Determine "magic number" or remove it
			event.confidence = -lnOneMinusChisqCdf(event.snr * 0.62, event.chisq_dof * 0.62)

			# This is done here so that the current PSD is used rather than what
			# might be there when the triggers are actually output
			utils.compute_amplitude(event, self.psd)

			event.snr = numpy.sqrt(event.snr / event.chisq_dof - 1)

			# Reassign IDs since they won't be unique
			event.event_id = self.triggers.get_next_id()
			event.process_id = self.process.process_id
			self.event_number += 1

			# If we're using a different units system, adjust back to SI
			# Readjust start time for units
			event.set_start(event.get_start() / self.units)
			event.set_peak(event.get_peak() / self.units)
			event.duration /= self.units

			self.triggers.append(event)

		# Update the timestamps which tell us how far along in the trigger
		# streams we are
		buf_ts = buf.timestamp*1e-9 / self.units
		buf_dur = buf.duration*1e-9 / self.units
		self.stop = (buf_ts + buf_dur)

		# Check if clustering reduces the amount of events
		if len(self.triggers) >= self.max_events and self._clustering:
			self.process_triggers(cluster_passes=1)

		# We use the buffer timestamp here, since it's always guaranteed to be
		# the earliest available buffer, so we guarantee that the span of
		# triggers is always greater than file stride duration
		if buf_ts - self.time_since_dump > self.dump_frequency or len(self.triggers) >= self.max_events:
			trigseg = segment(LIGOTimeGPS(self.time_since_dump), LIGOTimeGPS(buf_ts))
			outseg = segment(LIGOTimeGPS(self.time_since_dump), LIGOTimeGPS(self.time_since_dump + self.dump_frequency))
			outseg = trigseg if abs(trigseg) < abs(outseg) else outseg

			subdir = utils.append_formatted_output_path(self.outdirfmt, self, mkdir=False)
			subdir = os.path.join(self.outdir, subdir)
			if not os.path.exists(subdir):
				self.lock.acquire()
				os.makedirs(subdir)
				self.lock.release()
			fname = utils.make_cache_parseable_name(
				inst = self.inst,
				tag = self.channel,
				start = float(outseg[0]),
				stop = float(outseg[1]),
				ext = "xml.gz",
				dir = subdir
			)

			if self._clustering:
				# Do the final clustering
				self.process_triggers(cluster_passes = True)

				# Final check on clustered SNR
				for i in range(len(self.triggers))[::-1]:
					if self.snr_thresh and self.triggers[i].snr < self.snr_thresh:
						del self.triggers[i]

			self.write_triggers(filename = fname, seg = outseg)
			self.time_since_dump = float(outseg[1])

	def process_triggers(self, cluster_passes=0):
		"""
		Cluster triggers with a varying number of cluster passes. The cluster parameter controls how many passes of the clustering routine should be performed, with True being a special value indicating "as many as neccessary".
		FIXME: Dear god, the logic in this function is twisted. Rework.
		"""

		# Avoid all the temporary writing and the such
		if cluster_passes == 0 or not self._clustering:
			return

		full = cluster_passes is True
		if full:
			cluster_passes = 1
		# Pipe down unless its important
		verbose = self.verbose and full
		changed = True
		ligolw_bucluster.add_ms_columns_to_table(self.triggers)
		off = ligolw_bucluster.ExcessPowerPreFunc(self.triggers)
		while changed and self._clustering:
			changed = snglcluster.cluster_events( 
				events = self.triggers,
				testfunc = ligolw_bucluster.ExcessPowerTestFunc,
				clusterfunc = ligolw_bucluster.ExcessPowerClusterFunc,
				sortfunc = ligolw_bucluster.ExcessPowerSortFunc,
				bailoutfunc = ligolw_bucluster.ExcessPowerBailoutFunc,
				verbose = verbose
			)
			# If full clustering is on, ignore the number of cluster_passes
			if not full:
				cluster_passes -= 1
			# If we've reached the number of requested passes, break out
			if cluster_passes <= 0: 
				break
		ligolw_bucluster.ExcessPowerPostFunc(self.triggers, off)

	# FIXME: Remove flush argument, it serves no purpose
	def write_triggers(self, filename, flush=False, seg=None):

		if not self.output:
			return

		output = ligolw.Document()
		output.appendChild(ligolw.LIGO_LW())

		requested_segment = seg or segment(
			LIGOTimeGPS(self.time_since_dump), 
			LIGOTimeGPS(self.stop)
		)
		print >>sys.stderr, "req seg: %s" % str(requested_segment)

		# If we include start up time, indicate it in the search summary
		self.whiten_seg = segment( 
			LIGOTimeGPS(self.start), 
			LIGOTimeGPS(self.start + self.whitener_offset)
		)

		analysis_segment = utils.determine_segment_with_whitening( 
			requested_segment, self.whiten_seg 
		)
		print >>sys.stderr, "ana seg: %s" % str(analysis_segment)

		process = self.make_process_tables(None, output)

		# Append only triggers in requested segment
		outtable = EPHandler.make_output_table(self._clustering)
		for i in list(range(len(self.triggers)))[::-1]:
			# FIXME: Less than here rather than a check for being in the segment
			# This is because triggers can arrive "late" and thus not be put in
			# the proper file span. This might be a bug in the AppSync.
			if self.triggers[i].get_peak() < analysis_segment[1]:
				outtable.append(self.triggers[i])	
				del self.triggers[i]
		output.childNodes[0].appendChild(outtable)

		ligolw_search_summary.append_search_summary(output, process, lalwrapper_cvs_tag=None, lal_cvs_tag=None, inseg=requested_segment)
		search_sum = lsctables.SearchSummaryTable.get_table(output)
		# TODO: This shouldn't set every one of them in case we reuse XML 
		# documents later
		for row in search_sum:
			row.set_out(analysis_segment)

		# The second condition corresponds to a sentinel that I think
		# is a bug in the segment class. If start > end, then it simply
		# reverses the order of and creates a segment with a 
		# non-negative duration. What can happen is the gate gets 
		# ahead of the trigger generator by enough that the "analysis"
		# would end before the next segment. The second condition
		# prevents that segment from being added to the outgoing search
		# summary.
		cur_seg = None
		if self.current_segment is not None and float(self.current_segment[0]) <= analysis_segment[1]:
			# add the current segment
			cur_seg = segment(self.current_segment[0], LIGOTimeGPS(analysis_segment[1]))
			self.seglist["state"].append(cur_seg)
			# TODO: send the new time to handle_segment instead
			self.current_segment = segment(cur_seg[1], PosInfinity)

		# Write segments
		llwseg = ligolw_segments.LigolwSegments(output)
		# FIXME: Better names and comments?
		llwseg.insert_from_segmentlistdict(self.seglist, u"gstlal_excesspower segments", comment="gstlal_excesspower segments")

		llwseg.finalize(process)

		self.seglist.clear()
		self.seglist["state"] = segmentlist([])

		if self.stop < 0: # indication that we're quitting with no output
			return

		print >>sys.stderr, "Outputting triggers for %s\n" % str(requested_segment)

		# write the new distribution stats to disk
		self.lock.acquire()
		ligolw_utils.write_filename(output, filename, verbose = self.verbose, gz = (filename or "stdout").endswith(".gz"), trap_signals = None)
		self.lock.release()

		# Keep track of the output files we make for later convience
		if self.output_cache_name is not None:
			self.output_cache.append(
				CacheEntry(
					self.inst, self.channel + "_excesspower",
					analysis_segment,
					("file://localhost" + os.path.abspath(filename))
				)
			)

		# Keeping statistics about event rates
		# FIXME: This needs to be moved up before the trigger dumping
		if self.channel_monitoring:
			self.stats.add_events(self.triggers, cur_seg)
			self.stats.normalize()
			stat_json = {}
			stat_json["current_seg"] = [float(t) for t in (cur_seg or analysis_segment)] 
			stat_json["offsource_time"] = sum([float(abs(t)) for s in self.stats.offsource.keys()])
			stat_json["psd_last_change_percent"] = self.psd_change
			stat_json["psd_power"] = self.psd_power
			ontime = float(abs(segmentlist(self.stats.onsource.keys())))
			erate = 0
			for sbt in self.stats.onsource.values(): 
				erate += len(sbt)
			if ontime > 0:
				erate /= ontime
				stat_json["event_rate"] = erate
			else:
				stat_json["event_rate"] = float('NaN')

			rates = self.stats.event_rate()
			stat_json["event_rates"] = list(rates)
			esig = self.stats.event_significance()
			stat_json["event_significance"] = list(esig)
			if self.verbose:
				print >>sys.stderr, "Event rate in current segment: %g" % erate
				print >>sys.stderr, "Event significance:\nrank\t\tsig"
				for (snr, sig) in esig:
					print >>sys.stderr, "%4.2f\t\t%4.2f" % (snr, sig)

			self.lock.acquire()
			jsonf = open("%s-%s-channel_mon.json" % (self.inst, self.channel), "w")
			print >>jsonf, json.dumps(stat_json)
			jsonf.close()
			self.lock.release()

		if flush: 
			self.triggers = EPHandler.make_output_table(self._clustering)
		
	def shutdown(self, signum, frame):
		"""
		Method called to flush buffers and shutdown the pipeline.
		"""
		print >>sys.stderr, "Caught signal, signal received, if any: " + str(signum)
		self.pipeline.set_state(gst.STATE_PAUSED)
		bus = self.pipeline.get_bus()
		bus.post(gst.message_new_eos(self.pipeline))
		self.pipeline.set_state(gst.STATE_NULL)
		self.mainloop.quit()

		if signum is not None:
			return

		subdir = "./"
		if self.outdirfmt is not None:
			subdir = utils.append_formatted_output_path(self.outdirfmt, self)
			subdir = os.path.join(self.outdir, subdir)
		if not os.path.exists(subdir):
			self.lock.acquire()
			os.makedirs(subdir)
			self.lock.release()

		outfile = utils.make_cache_parseable_name(
			inst = self.inst,	
			tag = self.channel,
			start = self.time_since_dump,
			stop = self.stop,
			ext = "xml.gz",
			dir = subdir
		)
		self.process_triggers(self._clustering)
		self.write_triggers(filename = outfile)

		if self.output_cache:
			outfile = self.output_cache_name or utils.make_cache_parseable_name(
				inst = self.inst,	
				tag = self.channel,
				start = self.start,
				stop = self.stop,
				ext = "cache",
				dir = self.outdir
			)
			self.lock.acquire()
			self.output_cache.tofile(file(outfile, "w"))
			self.lock.release()

		self.destroy_filter_xml()
		if os.path.exists(self.tempdir):
			os.removedirs(self.tempdir)

def mknxyfdsink(pipeline, src, fd, segment = None, units = utils.EXCESSPOWER_UNIT_SCALE['Hz']):
    if segment is not None:
        elem = pipeparts.mkgeneric(pipeline, src, "lal_nxydump", start_time = int(segment[0].ns()*units), stop_time = int(segment[1].ns()*units))
    else:
        elem = pipeparts.mkgeneric(pipeline, src, "lal_nxydump")
    return pipeparts.mkgeneric(pipeline, elem, "fdsink", fd=fd, sync=False, async=False)

#
# =============================================================================
#
#                        Message Handler Methods
#
# =============================================================================
#

def on_psd_change(elem, pspec, handler, drop_time):
    """
    Get the PSD object and signal the handler to rebuild everything if the spectrum has changed appreciably from the PSD which was used to rebuild the filters originally.
    """
    if handler.verbose:
        print >>sys.stderr, "Intercepted spectrum signal."

    # Get the new one
    # FIXME: Reincorpate the kwargs
    new_psd = laltypes.REAL8FrequencySeries(name = "PSD", f0 = 0.0, deltaF = elem.get_property("delta-f"), data = numpy.array(elem.get_property("mean-psd"))) #, epoch = laltypes.LIGOTimeGPS(0, message.timestamp), sampleUnits = laltypes.LALUnit(message.sample_units.strip()))
    handler.cur_psd = new_psd

    # Determine if the PSD has changed enough to warrant rebuilding the 
    # filter bank.
    handler.psd_power = sum(new_psd.data)*handler.psd.deltaF
    
    whitener_pos = elem.query_position(gst.FORMAT_TIME)[0]*1e-9

    # This will get triggered in two cases: the rate (and thus bin length)
    # has changed, or we had the default PSD in place previously.
    if len(new_psd.data) != len(handler.psd.data):
        if handler.verbose:

            print >>sys.stderr, "Different PSD lengths detected, automatically regenerating filters."
        handler.psd = new_psd
        handler.rebuild_everything()
        return
    else:
        # Poor man's coherence
        handler.psd_change = 2.0/len(handler.psd.data) * sum(abs(handler.psd.data-new_psd.data)/(new_psd.data+handler.psd.data))
    #psd_change = abs(handler.psd.data-new_psd.data)/(new_psd.data+handler.psd.data)
    #print >>sys.stderr , "PSD estimate: %g / %g (min: %g, max: %g)" % (sum(handler.psd.data),sum(new_psd.data), min(psd_change), max(psd_change))
    #print >>sys.stderr , "change estimate: %f" % handler.psd_change

    if abs(handler.psd_change) > handler.psd_change_thresh and whitener_pos - handler.start > 0.75*drop_time:
        if handler.verbose:
            print >> sys.stderr, "Processed signal, change estimate: %f, regenerating filters" % handler.psd_change
        handler.psd = new_psd
        handler.rebuild_everything()


def on_spec_corr_change(elem, pspec, handler):
    """
    Get the 2-point spectral correlation object and signal the handler to rebuild everything.
    """
    if handler.verbose:
        print >> sys.stderr, "Intercepted correlation signal."
    handler.spec_corr = elem.get_property("spectral-correlation")

    # If the spectrum correlation changes, rebuild everything
    if handler.psd is not None:
        handler.rebuild_everything()

# FIXME: Make into bin
def construct_excesspower_pipeline(pipeline, head, handler, scan_obj=None, drop_time=0, peak_fraction=None, disable_triggers=False, histogram_triggers=False, verbose=False):

    # Scan piece: Save the raw time series
    if handler.trigger_segment:
        head = pipeparts.mktee(pipeline, head)
        scan_obj.add_data_sink(pipeline, head, "time_series", "time")

    # Resample down to the requested rate
    head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head ), "audio/x-raw-float,rate=%d" % handler.rate)

    # Add reblock to break up data into smaller pieces
    head = pipeparts.mkreblock(pipeline, head, block_duration = 1*gst.SECOND)

    head = handler.whitener = pipeparts.mkwhiten(pipeline, head)
    # Amount of time to do a spectrum estimate over
    handler.whitener.set_property("fft-length", handler.fft_length) 

    # Ignore gaps in the whitener. Turning this off will induce wild triggers 
    # because the whitener will see sudden drops in the data (due to the gaps)
    handler.whitener.set_property("expand-gaps", True) 

    # Drop first PSD estimates until it settles
    if handler.psd_mode != 1:
        handler.whitener_offset = drop_time
        head = pipeparts.mkdrop(pipeline, head, int(drop_time*handler.rate))

    # Scan piece: Save the whitened time series
    if handler.trigger_segment:
        head = pipeparts.mktee(pipeline, head)
        scan_obj.add_data_sink(pipeline, head, "white_series", "time")

    head = pipeparts.mkqueue(pipeline, head, max_size_time = 1*gst.SECOND)

    if verbose:
        head = pipeparts.mkprogressreport(pipeline, head, "whitened stream")

    # excess power channel firbank
    head = pipeparts.mkfirbank(pipeline, head, time_domain=False, block_stride=handler.rate)

    # This function stores a reference to the FIR bank, creates the appropriate 
    # filters and sets the other options appropriately
    handler.add_firbank(head)
    nchannels = handler.filter_bank.shape[0]
    if verbose:
        print "FIR bank constructed with %d %f Hz channels" % (nchannels, handler.base_band)

    if verbose:
        head = pipeparts.mkprogressreport(pipeline, head, "FIR bank stream")

    # Scan piece: Save the filtered time series
    if handler.trigger_segment:
        postfirtee = head = pipeparts.mktee(pipeline, head)
        scan_obj.add_data_sink(pipeline, postfirtee, "filter_series", "time")

    # object to handle the synchronization of the appsinks
    # FIXME: Rework this
    def get_triggers_with_handler(elem):
        return handler.get_triggers(elem)
    appsync = pipeparts.AppSync(appsink_new_buffer = get_triggers_with_handler)

    # If the rate which would be set by the undersampler falls below one, we 
    # have to take steps to prevent this, as gstreamer can't handle this. The 
    # solution is to change the "units" of the rate. Ideally, this should be 
    # done much earlier in the pipeline (e.g. as the data comes out of the 
    # source), however, to avoid things like figuring out what that means for 
    # the FIR bank we change units here, and readjust appropriately in the 
    # trigger output.
    min_rate = 2 * handler.base_band

    unit = 'Hz'
    if min_rate < 1:
        print "Adjusting units to compensate for undersample rate falling below unity."
        # No, it's not factors of ten, but rates which aren't factors
        # of two are often tricky, thus if the rate is a factor of two, the 
        # units conversion won't change that.
        if min_rate > utils.EXCESSPOWER_UNIT_SCALE['mHz']:
            unit = 'mHz'
        elif min_rate > utils.EXCESSPOWER_UNIT_SCALE['uHz']:
            unit = 'uHz'
        elif min_rate > utils.EXCESSPOWER_UNIT_SCALE['nHz']:
            unit = 'nHz'
        else:
            sys.exit( "Requested undersampling rate would fall below 1 nHz." )
        # FIXME: No love for positive power of 10 units?

        handler.units = utils.EXCESSPOWER_UNIT_SCALE[unit]
        if handler.units != utils.EXCESSPOWER_UNIT_SCALE["Hz"]:
            head = pipeparts.mkgeneric(pipeline, head, "audioratefaker")
            head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float,rate=%d,channels=%d,width=64" % (handler.rate/handler.units, nchannels))

    postfirtee = pipeparts.mktee(pipeline, head)


    # First branch -- send fully sampled data to wider channels for processing
    nlevels = int(numpy.ceil(numpy.log2(nchannels))) 
    for res_level in range(0, min(handler.max_level, nlevels)):
        # New level bandwidth
        band = handler.base_band * 2**res_level

        # The undersample_rate for band = R/2 is => sample_rate (passthrough)
        orig_rate = 2 * band
        undersamp_rate = 2 * band / handler.units

        print "Undersampling rate for level %d: %f Hz -> %f %s" % (res_level, orig_rate, undersamp_rate, unit)

        head = postfirtee
        # queue up data
        head = pipeparts.mkqueue(pipeline, head, max_size_time = 1*gst.SECOND)

        head = pipeparts.mkgeneric(pipeline, head, "lal_audioundersample")
        head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float,rate=%d" % undersamp_rate)

        if verbose:
            head = pipeparts.mkprogressreport(pipeline, head, "Undersampled stream level %d" % res_level)

        # This converts N base band channels into M wider channels via the use
        # of a NxM matrix with M block diagonal elements containing the proper 
        # renormalization per row
        head = matmixer = pipeparts.mkmatrixmixer(pipeline, head)
        handler.add_matmixer(matmixer, res_level)

        if verbose:
            head = pipeparts.mkprogressreport(pipeline, head, "post matrix mixer %d" % res_level)

        # Square the samples to get energies to sum over
        # FIXME: Put the power in lal_mean
        head = pipeparts.mkgeneric(pipeline, head, "pow", exponent=2)

        if verbose:
            head = pipeparts.mkprogressreport(pipeline, head, "Energy stream level %d" % res_level)

        # Second branch -- duration
        max_samp = int(handler.max_duration*undersamp_rate*handler.units)

        # If the user requests a maximum DOF, we use that instead
        if handler.max_dof is not None:
            max_samp = handler.max_dof

        if max_samp < 2 and res_level == handler.max_level:
            sys.exit("The duration for the largest tile is smaller than a two degree of freedom tile. Try increasing the requested maximum tile duration or maximum DOF requirement.")
        elif max_samp < 2:
            print "Further resolution levels would result in tiles for which the maximum duration (%f) would not have enough DOF (2). Skipping this levels." % handler.max_duration
            continue
        print "Can sum up to %s degress of freedom in powers of two for this resolution level." % max_samp

        # samples to sum -- two is min number
        ndof = 2

        head = pipeparts.mktee(pipeline, head)
        while ndof <= max_samp:
            if handler.fix_dof is not None and ndof != handler.fix_dof:
                ndof <<= 1
                continue

            if ndof/undersamp_rate > handler.max_duration:
                break

            # Scan piece: Save the summed square NDOF=1 stream
            if handler.trigger_segment:
                scan_obj.add_data_sink(pipeline, head, "sq_sum_series_level_%d_dof_1" % res_level, "time", handler.units)

            if verbose:
                print "Resolution level %d, DOFs: %d" % (res_level, ndof)

            # Multi channel FIR filter -- used to add together frequency bands 
            # into tiles
            # FIXME: Use units / samples here
            durtee = pipeparts.mkqueue(pipeline, head, max_size_time = 1*gst.SECOND)
            durtee = pipeparts.mkmean(pipeline, durtee, n=ndof, type=2, moment=1)

            # Scan piece: Save the summed square NDOF=2 stream
            if handler.trigger_segment:
                durtee = pipeparts.mktee(pipeline, durtee)
                scan_obj.add_data_sink(pipeline, durtee, "sq_sum_series_level_%d_dof_%d" % (res_level, ndof), "time", handler.units)

            if verbose:
                durtee = pipeparts.mkprogressreport(pipeline, durtee, "After energy summation resolution level %d, %d DOF" % (res_level, ndof))

            if disable_triggers:
                pipeparts.mkfakesink(pipeline, durtee)
                ndof = ndof << 1
                continue

            # FIXME: This never seems to work, but it could be very useful as a 
            # diagnostic
            if histogram_triggers:
                durtee = pipeparts.mkqueue(pipeline, durtee)
                durtee = tmptee = pipeparts.mktee(pipeline, durtee)
                tmptee = pipeparts.mkhistogram(pipeline, tmptee)
                #tmptee = pipeparts.mkcolorspace(pipeline, tmptee)
                #pipeparts.mkgeneric(pipeline, tmptee, "autovideosink", filter_caps=gst.caps_from_string("video/x-raw-rgb"))
                pipeparts.mkogmvideosink(pipeline, tmptee, "test_%d_%d.ogm" % (res_level, ndof))

            # FIXME: one the downstream elements is passing buffers marked
            # as disconts. This resets the triggergen element, and the
            # timestamps are very messed up as a consequence. This is a
            # temporary fix until I can figure out which element is culprit
            # and why its only happening with fixed PSDs
            if handler.psd_mode == 1:
                durtee = pipeparts.mknofakedisconts(pipeline, durtee)

            # Downsample the SNR stream
            #durtee = pipeparts.mkresample(pipeline, durtee)
            #snr_rate = int(undersamp_rate/max(1, ndof*(peak_fraction or 0)))
            #durtee = pipeparts.mkcapsfilter(pipeline, durtee, "audio/x-raw-float,rate=%d" % snr_rate)

            # Trigger generator
            # Determine the SNR threshold for this trigger generator
            snr_thresh = utils.determine_thresh_from_fap(handler.fap, ndof)**2
            if verbose:
                print "SNR threshold for level %d, ndof %d: %f" % (res_level, ndof, snr_thresh)
            durtee = pipeparts.mkbursttriggergen(pipeline, durtee, n=int((peak_fraction or 0) * ndof), bank_filename=handler.build_filter_xml(res_level, ndof, verbose=verbose), snr_thresh=snr_thresh)

            if verbose:
                durtee = pipeparts.mkprogressreport(pipeline, durtee, "Trigger generator resolution level %d, %d DOF" % (res_level, ndof))
                
            # Funnel the triggers for this subbranch to the appsink
            # FIXME: Why doesn't this negotiate the caps properly?
            #appsync.add_sink( pipeline, pipeparts.mkqueue(pipeline, durtee), caps = gst.Caps("application/x-lal-snglburst") )
            appsync.add_sink(pipeline, pipeparts.mkqueue(pipeline, durtee, max_size_buffers = 10))

            ndof <<= 1
            if ndof > max_samp:
                break

    return handler

def stream_tfmap_video(pipeline, head, handler, filename=None, split_on=None, snr_max=None, history=4, framerate=5):
	"""
	Stream the time frequency channel map to a video source. If filename is None and split_on is None (the default), then the pipeline will attempt to stream to a desktop based (xvimagesink or equivalent) video sink. If filename is not None, but no splitting behavior is specified, video will be encoded and saved to the filename plus ".ogg" in Ogg Vorbis format. If split_on is specified to be 'keyframe', then the encoded video will be split between multiple files based on the keyframes being emitted by the ogg muxer. If no file name is specifed a default will be used, otherwise, an index and ".ogg" will be appended to the file name. Specifying amp_max will set the top of the colorscale for the amplitude SNR, the default is 10. History is the amount of time to retain in the video buffer (in seconds), the default is 4. The frame rate is the number of frames per second to output in the video stream.
	"""

	if snr_max is None:
		snr_max = 10 # arbitrary
		z_autoscale = True 
	# Tee off the amplitude stream
	#y_autoscale = True,
	#y_min = handler.flow,
	#y_max = handler.fhigh,
	head = chtee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkqueue(pipeline, head)
	head = pipeparts.mkgeneric(pipeline, head, "cairovis_waterfall", title = "TF map %s:%s" % (handler.inst, handler.channel), z_autoscale = z_autoscale, z_min = 0, z_max = snr_max, z_label = "SNR", y_data_autoscale = False, y_data_min = handler.flow, y_data_max = handler.fhigh, y_label = "frequency (Hz)", x_label = "time (s)", colormap = "jet", colorbar = True, history = gst.SECOND*history)

	# Do some format conversion
	head = pipeparts.mkcapsfilter(pipeline, head, "video/x-raw-rgb,framerate=%d/1" % framerate)
	head = pipeparts.mkprogressreport(pipeline, head, "video sink")

	# TODO: Explore using different "next file" mechanisms
	if split_on == "keyframe":

		# Muxer
		head = pipeparts.mkcolorspace(pipeline, head)
		head = pipeparts.mkcapsfilter(pipeline, head, "video/x-raw-yuv,framerate=5/1")
		head = pipeparts.mkoggmux(pipeline, pipeparts.mktheoraenc(pipeline, head))

		if filename is None: 
			filename = handler.inst + "_tfmap_%d.ogg"
		else: 
			filename = filename + "_%d.ogg"

		print >>sys.stderr, "Streaming TF maps to %s\n" % filename
		pipeparts.mkgeneric( pipeline, head, "multifilesink", next_file = 2, location = filename, sync = False, async = False)

	elif filename is not None:
		# Muxer
		head = pipeparts.mkcolorspace(pipeline, head)
		head = pipeparts.mkcapsfilter(pipeline, head, "video/x-raw-yuv,framerate=5/1")
		head = pipeparts.mkoggmux(pipeline, pipeparts.mktheoraenc(pipeline, head))
		filename = filename + ".ogg"
		pipeparts.mkfilesink(pipeline, head, filename)

	else: # No filename and no splitting options means stream to desktop
		pipeparts.mkgeneric(pipeline, head, "autovideosink", filter_caps=gst.caps_from_string("video/x-raw-rgb"))

	return chtee
