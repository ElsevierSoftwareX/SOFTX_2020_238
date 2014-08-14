#!/usr/bin/env python
#
# Copyright (C) 2012 Chris Pankow
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
import signal
import glob
import threading
import json
import tempfile
import shutil
import types

import numpy

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst

from gstlal.simplehandler import Handler
from gstlal.reference_psd import write_psd, read_psd_xmldoc

import gstlal.excesspower as ep

from glue.ligolw import ligolw, lsctables, table, ilwd
class ContentHandler(ligolw.LIGOLWContentHandler): 
	pass 
lsctables.use_in(ligolw.LIGOLWContentHandler)

from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import search_summary as ligolw_search_summary

from glue.segments import segment, segmentlist, segmentlistdict, PosInfinity
from glue import segmentsUtils
from glue.lal import Cache, CacheEntry

from pylal import snglcluster
from pylal import ligolw_bucluster

from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries
from pylal.xlal.datatypes.lalunit import LALUnit
from pylal.xlal.datatypes.snglburst import SnglBurst
from pylal.xlal.datatypes.snglburst import from_buffer as sngl_bursts_from_buffer

import lalburst

#
# =============================================================================
#
#                                Handler Class
#
# =============================================================================
#

class EPHandler( Handler ):
	"""
	Handler class for the excess power pipeline. Keeps various bits of information that the pipeline emits and consumes. This is also in charge of signalling the rebuild of various matrices and vectors needed by the pipeline.
	"""
	def __init__( self, mainloop, pipeline ):

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
		self.units = ep.EXCESSPOWER_UNIT_SCALE['Hz']

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
		self.outdir = "./"
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

		self.bus = pipeline.get_bus()
		self.bus.add_signal_watch()
		self.bus.connect( "message", self.process_message )

		self.pipeline = pipeline

		self.verbose = False
		self._clustering = False
		self.channel_monitoring = False
		self.stats = ep.SBStats()

		super(type(self), self).__init__(mainloop, pipeline)

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

	def handle_segment( self, elem, timestamp, segment_type ):
		"""
		Process state changes from the state vector mechanism.
		"""
		if segment_type == "on":
			self.current_segment = segment( LIGOTimeGPS(timestamp) / 1e9, PosInfinity )
			#segmentsUtils.tosegwizard( sys.stdout, self.seglist["state"] )
			if self.verbose:
				print >>sys.stderr, "Starting segment #%d: %.9f" % (len(self.seglist["state"]), self.current_segment[0])
		elif segment_type == "off":
			if self.current_segment is None: 
				print >>sys.stderr, "Got a message to end a segment, but no current segment exists. Ignoring."
				return
			self.seglist["state"].append(
				segment( self.current_segment[0], LIGOTimeGPS(timestamp / 1e9) )
			)
			# Make it very clear we don't have a segment currently
			self.current_segment = None
			if self.verbose:
				print >>sys.stderr, "Ending segment #%d: %s" % (len(self.seglist["state"])-1, str(self.seglist["state"][-1]))
		else:
			print >>sys.stderr, "Unrecognized state change, ignoring."

	def process_message( self, bus, message ):
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
			if self.psd_mode == 1 and self.psd is not None:
				if self.verbose:
					print >>sys.stderr, "Got tags message, fixing current PSD to whitener."
				self.whitener.set_property( "mean-psd", self.psd.data )
				self.whitener.set_property( "psd-mode", self.psd_mode ) # GSTLAL_PSDMODE_FIXED
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
				self.dump_psd( ts, self.cache_psd_dir )
			elif self.cache_psd is not None and self.cache_psd + self.last_psd_cache < ts:
				self.dump_psd( ts, self.cache_psd_dir )
				self.last_psd_cache = ts

	def dump_psd( self, timestamp, psddir="./" ):
		"""
		Dump the currently cached PSD to a LIGOLW XML file. If the psddir isn't specified, it defaults to the current directory.
		"""

		filename = ep.make_cache_parseable_name(
			inst = self.inst,
			tag = self.channel + "_PSD",
			start = round(timestamp) - 40, # PSD history length
			stop = round(timestamp),
			ext = "xml.gz",
			dir = psddir
		)

		write_psd( filename, { self.inst: self.cur_psd } )

	def add_firbank( self, firbank ):
		"""
		Set the main base band FIR bank, and build the FIR matrix.
		"""
		firbank.set_property( "fir-matrix", self.rebuild_filter() )
		# Impose a latency since we've advanced the filter in the 
		# generation step. See build_filter in the excesspower library
		firbank.set_property( "latency", len(firbank.get_property("fir_matrix")[0])/2 )
		self.firbank = firbank

	def add_matmixer( self, mm, res_level ):
		self.mmixers[ res_level ] = mm
		self.rebuild_matrix_mixers( res_level )

	@staticmethod
	def build_default_psd( rate, df, fhigh ):
		"""
		Builds a dummy PSD to use until we get the right one.
		"""
		psd = REAL8FrequencySeries()
		psd.deltaF = df
		psd.sampleUnits = LALUnit( "s strain^2" )
		psd.data = numpy.ones( int(rate/2/df) + 1 ) 
		psd.f0 = 0
		return psd

	@staticmethod
	def build_default_correlation( rate ):
		"""
		Builds a Kronecker delta correlation series for k, k'.
		"""
		# FIXME: rate isn't right
		corr = numpy.zeros(rate + 1)
		corr[0] = 1
		return corr

	def rebuild_matrix_mixers( self, res_level = None ):
		"""
		Rebuilds the matrix mixer matrices from the coefficients calculated in rebuild_chan_mix_matrix and assigns them to their proper element.
		"""
		for i, mm in self.mmixers.iteritems():
			if res_level is not None and res_level != i: 
				continue

			nchannels = self.filter_bank.shape[0]
			self.chan_matrix[i] = ep.build_wide_filter_norm( 
				corr = self.spec_corr, 
				freq_filters = self.freq_filter_bank,
				level = i,
				frequency_overlap = self.frequency_overlap,
				band = self.base_band
			)
			cmatrix = ep.build_chan_matrix( 
				nchannels = nchannels,
				frequency_overlap = self.frequency_overlap,
				up_factor = i,
				norm = self.chan_matrix[i]
			) 
			# DEBUG: Uncommet to get matrix mixer elements
			#cmatrix.tofile( open("matrix_level_%d" % i, "w") )
			mm.set_property( "matrix", cmatrix )

	def rebuild_filter( self ):
		"""
		Calling this function rebuilds the filter FIR banks and assigns them to their proper element. This is normally called when the PSD or spectrum correlation changes.
		"""
		if not self.filter_xml.has_key((0,1)):
			self.build_filter_xml( res_level = 0, ndof = 1 )

		self.filter_bank, self.freq_filter_bank = ep.build_filter_from_xml( 
			self.filter_xml[(0,1)],
			self.psd,
			self.spec_corr
		)
		return self.filter_bank

	def build_filter_xml( self, res_level, ndof=1, loc="", verbose=False ):
		"""
		Calls the EP library to create a XML of sngl_burst tables representing the filter banks. At the moment, this dumps them to the current directory, but this can be changed by supplying the 'loc' argument. The written filename is returned for easy use by the trigger generator.
		"""
		self.filter_xml[(res_level, ndof)] = ep.create_bank_xml(
			self.flow,
			self.fhigh,
			self.base_band*2**res_level,
			1.0 / (2*self.base_band*2**res_level), # resolution level starts from 0
			res_level,
			ndof,
			self.frequency_overlap,
			self.inst
		)

		# Store the filter name so we can destroy it later
		output = "%sgstlal_excesspower_bank_%s_%s_level_%d_%d.xml" % (loc, self.inst, self.channel, res_level, ndof)
		self.filter_xml[output] = self.filter_xml[(res_level, ndof)]

		# Write it
		self.lock.acquire()
		utils.write_filename( self.filter_xml[output], 
			output, 
			verbose = verbose,
		    gz = (output or "stdout").endswith(".gz") )
		self.lock.release()

		# Just get the table we want
		self.filter_xml[(res_level, ndof)] = table.get_table(self.filter_xml[(res_level, ndof)], lsctables.SnglBurstTable.tableName )
		return output

	def destroy_filter_xml( self, loc="" ):
		"""
		Remove the filter XML files.
		"""
		for f, v in self.filter_xml.iteritems():
			if os.path.exists( str(f) ):
				os.remove( f )

	def rebuild_chan_mix_matrix( self ):
		"""
		Calling this function rebuilds the matrix mixer coefficients for higher resolution components. This is normally called when the PSD or spectrum correlation changes.
		"""
		self.chan_matrix = ep.build_wide_filter_norm( 
			corr = self.spec_corr, 
			freq_filters = self.freq_filter_bank,
			level = self.max_level,
			frequency_overlap = self.frequency_overlap
		)
		return self.chan_matrix

	def rebuild_everything( self ):
		"""
		Top-level function to handle the asynchronous updating of FIR banks and matrix mixer elements.
		"""
		# Rebuild filter bank and hand it off to the FIR element
		if self.verbose:
			print >>sys.stderr, "Rebuilding FIR bank"
		self.firbank.set_property( "fir_matrix", self.rebuild_filter() )
		latency = len(self.firbank.get_property("fir_matrix")[0])/2+1 
		self.firbank.set_property( "latency", latency )
		if self.verbose:
			print >>sys.stderr, "New filter latency %d (%f s)" % (latency, latency/float(self.rate))

		if self.verbose:
			print >>sys.stderr, "Rebuilding matrix mixer"
		#self.rebuild_chan_mix_matrix()
		# Rebuild the matrix mixer with new normalization coefficients
		self.rebuild_matrix_mixers()
		if self.verbose:
			print >>sys.stderr, "...done."

	def make_process_tables( self, options=None, xmldoc=None ):
		"""
		Create a process and process_params table for use in the output document. If the options parameter is passed, the process_params table will be created from it. If the output xmldoc is provided, it will register the pipeline with the document. Returns the process and process_params table, however, note that the process table will be empty if no options are provided.
		"""
		if options:
			self.process_params = vars( options )

		if self.process_params is None:
			print >>sys.stderr, "WARNING: Options have not yet been set in the handler. Process and ProcessParams may not be constructed properly. Call handler.make_process_tables() with the options argument to set command line options."
			return lsctables.New(lsctables.ProcessTable)

		if self.process is None:
			xmldoc = ligolw.Document() # dummy document
			xmldoc.appendChild(ligolw.LIGO_LW())
			self.process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_excesspower", self.process_params)
			self.process.set_ifos( [self.inst] )
		elif xmldoc and self.process is not None:
			# This branch is here to prevent the register_to_xmldoc method from
			# creating a new process id for each call to this function, since
			# it can be called several times during one run of the same pipeline
			# Note that this is basically the same code as register_to_xmldoc
			# it just preserves the process row
			ptable = lsctables.New(lsctables.ProcessTable)
			ptable.append( self.process )
			xmldoc.childNodes[0].appendChild( ptable )
			ligolw_process.append_process_params( xmldoc, 
				self.process, 
				ligolw_process.process_params_from_dict( self.process_params )
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

		buffer = appsink.emit("pull-buffer")

		if not self.output:
			del buffer
			return # We don't want event information

		# TODO: Can I set units here on the buffer fields, avoid changing the 
		# triggers themslves and *not* screw up the underlying framework?

		# What comes out of the sngl_bursts_from_buffer is a
		# pylal.xlal.datatypes.snglburst.SnglBurst object. It does not have
		# all the trappings of its glue.ligolw.lsctables cousin, so we
		# convert it here first
		for event in [convert_sngl_burst(sb, self.triggers) for sb in sngl_bursts_from_buffer(buffer)]:

			# FIXME: Determine "magic number" or remove it
			event.confidence = -lalburst.lnOneMinusChisqCdf(event.snr * 0.62, event.chisq_dof * 0.62)

			# This is done here so that the current PSD is used rather than what
			# might be there when the triggers are actually output
			ep.compute_amplitude(event, self.psd)

			event.snr = numpy.sqrt(event.snr / event.chisq_dof - 1)

			# Reassign IDs since they won't be unique
			# FIXME: Use get_next_id
			#event.event_id = ilwd.ilwdchar("sngl_burst:event_id:%d" % self.event_number)
			event.event_id = self.triggers.get_next_id()
			event.process_id = self.process.process_id
			self.event_number += 1

			# If we're using a different units system, adjust back to SI
			event.duration *= self.units
			# Readjust start time for units
			# FIXME: Is self.start already in the 'new' unit system?
			# If so, we can delete these comments, otherwise, uncomment
			#event.start_time -= self.start
			event.start_time /= self.units
			event.peak_time /= self.units
			#event.start_time += self.start
			# FIXME: Readjust frequencies? I don't think so, the filter tables are
			# created 'independently and thus don't see the new units
			#event.bandwidth /= self.units

			self.triggers.append(event)

		# Update the timestamps which tell us how far along in the trigger
		# streams we are
		# TODO: Why does the buf_dur need unit conversion, but not the timestamp
		#buf_ts = (buffer.timestamp*1e-9 - self.start) / self.units + self.start
		buf_ts = buffer.timestamp*1e-9 
		buf_dur = buffer.duration*1e-9 / self.units
		self.stop = (buf_ts + buf_dur)

		del buffer

		# Check if clustering reduces the amount of events
		if len(self.triggers) >= self.max_events and self._clustering:
			self.process_triggers(cluster_passes=1)

		# We use the buffer timestamp here, since it's always guaranteed to be
		# the earliest available buffer, so we guarantee that the span of
		# triggers is always greater than file stride duration
		if buf_ts - self.time_since_dump > self.dump_frequency or len(self.triggers) >= self.max_events:
			trigseg = segment( LIGOTimeGPS(self.time_since_dump), LIGOTimeGPS(buf_ts) )
			outseg = segment( LIGOTimeGPS(self.time_since_dump), LIGOTimeGPS(self.time_since_dump + self.dump_frequency) )
			outseg = trigseg if abs(trigseg) < abs(outseg) else outseg

			subdir = ep.append_formatted_output_path( self.outdirfmt, self, mkdir=False )
			subdir = os.path.join(self.outdir, subdir)
			if not os.path.exists(subdir):
				self.lock.acquire()
				os.makedirs(subdir)
				self.lock.release()
			fname = ep.make_cache_parseable_name(
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

			# Final check on SNR
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

	def write_triggers( self, filename, flush=True, seg=None ):

		if not self.output:
			return

		output = ligolw.Document()
		output.appendChild(ligolw.LIGO_LW())

		requested_segment = seg or segment(
			LIGOTimeGPS( self.time_since_dump ), 
			LIGOTimeGPS( self.stop )
		)
		print >>sys.stderr, "req seg: %s" % str(requested_segment)

		# If we include start up time, indicate it in the search summary
		self.whiten_seg = segment( 
			LIGOTimeGPS(self.start), 
			LIGOTimeGPS(self.start + self.whitener_offset)
		)

		analysis_segment = ep.determine_segment_with_whitening( 
			requested_segment, self.whiten_seg 
		)
		print >>sys.stderr, "ana seg: %s" % str(analysis_segment)

		process = self.make_process_tables( None, output )

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

		ligolw_search_summary.append_search_summary( output, process, lalwrapper_cvs_tag=None, lal_cvs_tag=None, inseg=requested_segment )
		search_sum = lsctables.table.get_table( output, lsctables.SearchSummaryTable.tableName )
		# TODO: This shouldn't set every one of them in case we reuse XML 
		# documents later
		for row in search_sum:
			row.set_out( analysis_segment )

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
			cur_seg = segment( self.current_segment[0], LIGOTimeGPS(analysis_segment[1]) )
			self.seglist["state"].append( cur_seg )
			# TODO: send the new time to handle_segment instead
			self.current_segment = segment( cur_seg[1], PosInfinity )

		# Write segments
		llwseg = ligolw_segments.LigolwSegments( output )
		# FIXME: Better names and comments?
		llwseg.insert_from_segmentlistdict( self.seglist, u"gstlal_excesspower segments", comment="gstlal_excesspower segments" )

		llwseg.finalize(process)

		# FIXME: We should be careful to not fragment segments across output too
		# much
		self.seglist.clear()

		self.seglist["state"] = segmentlist([])

		if self.stop < 0: # indication that we're quitting with no output
			return

		# TODO: replace cbc filter table with our own
		#cbc_filter_table = lsctables.getTablesByType( output, lsctables.FilterTable )[0]
		#ep_filter_table = lsctables.getTablesByType( self.filter_xml, lsctables.FilterTable )[0]
		#output.replaceChild( ep_filter_table, cbc_filter_table )
		print >>sys.stderr, "Outputting triggers for %s\n" % str(requested_segment)

		# write the new distribution stats to disk
		self.lock.acquire()
		utils.write_filename(output, filename, verbose = self.verbose, gz = (filename or "stdout").endswith(".gz"), trap_signals = None)
		self.lock.release()

		# Keep track of the output files we make for later convience
		if self.output_cache is not None:
			self.output_cache.append(
				CacheEntry(
					self.inst, self.channel + "_excesspower",
					analysis_segment,
					("file://localhost" + os.path.abspath(filename))
				)
			)

		if self.output_cache_name is not None:	
			self.lock.acquire()
			# This is to ensure that multiple *processes* don't
			# write to the file simulateanously, and thereby
			# invalidating the cache
			counter, lockf = 100, None
			while counter > 0:
				try:
					lockf = os.open( self.output_cache_name + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY )
				except OSError:
					counter -= 1
				break

			if lockf is not None:
				self.output_cache.tofile( file(self.output_cache_name, "a") )
				os.close(lockf)
				os.remove( self.output_cache_name + ".lock" )
				
			self.lock.release()

		# Keeping statistics about event rates
		if self.channel_monitoring:
			self.stats.add_events( self.triggers, cur_seg )
			self.stats.normalize()
			stat_json = {}
			stat_json["current_seg"] = [ float(t) for t in (cur_seg or analysis_segment) ] 
			stat_json["offsource_time"] = sum([ float(abs(t)) for s in self.stats.offsource.keys() ])
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
			jsonf = open( "%s-%s-channel_mon.json" % (self.inst, self.channel), "w" )
			print >>jsonf, json.dumps( stat_json )
			jsonf.close()
			self.lock.release()

		if flush: 
			self.triggers = EPHandler.make_output_table(self._clustering)
		
	def shutdown( self, signum, frame ):
		"""
		Method called to flush buffers and shutdown the pipeline.
		"""
		print >>sys.stderr, "Caught signal, signal received, if any: " + str(signum)
		self.pipeline.set_state( gst.STATE_PAUSED )
		bus = self.pipeline.get_bus()
		bus.post(gst.message_new_eos(self.pipeline))
		self.pipeline.set_state( gst.STATE_NULL )
		self.mainloop.quit()

		print >>sys.stderr, "Please wait (don't ctrl+c) while I dump triggers to disk."

		subdir = "./"
		if self.outdirfmt is not None:
			subdir = ep.append_formatted_output_path( self.outdirfmt, self )
			subdir = os.path.join(self.outdir, subdir)
		if not os.path.exists( subdir ):
			self.lock.acquire()
			os.makedirs( subdir )
			self.lock.release()
		outfile = ep.make_cache_parseable_name(
			inst = self.inst,	
			tag = self.channel,
			start = self.time_since_dump,
			stop = self.stop,
			ext = "xml.gz",
			dir = subdir
		)
		self.process_triggers(self._clustering)

		# Final check on SNR
		for i in range(len(self.triggers))[::-1]:
			if self.snr_thresh and self.triggers[i].snr < self.snr_thresh:
				del self.triggers[i]

		self.write_triggers(filename = outfile)

		if self.output_cache:
			outfile = self.output_cache_name or ep.make_cache_parseable_name(
				inst = self.inst,	
				tag = self.channel,
				start = self.start,
				stop = self.stop,
				ext = "cache",
				dir = self.outdir
			)
			self.output_cache.tofile( file(outfile, "w") )

		self.destroy_filter_xml()

from gstlal import pipeparts
def mknxyfdsink(pipeline, src, fd, segment = None):
    if segment is not None:
        elem = pipeparts.mkgeneric(pipeline, src, "lal_nxydump", start_time = segment[0].ns(), stop_time = segment[1].ns())
    else:
        elem = pipeparts.mkgeneric(pipeline, src, "lal_nxydump")
    return pipeparts.mkgeneric(pipeline, elem, "fdsink", fd=fd, sync=False, async=False)

class EPScan(object):
	def __init__(self, scan_segment, low_freq, high_freq, base_band):
		self.serializer_dict = {}
		self.scan_segment = scan_segment
		self.bandwidth = segment(low_freq, high_freq)
		self.base_band = base_band

	def add_data_sink(self, pipeline, head, name, type):
		mknxyfdsink(pipeline,
			pipeparts.mkqueue(pipeline, head),
			self.get_tmp_fd(name, type),
			self.scan_segment
		)

	def get_tmp_fd(self, name, type):
		"""
        Create a temporary file and file descriptor, returning the descriptor... mostly for use with fdsink. Name is an internal identifier and 'write_out' will move the temporary file to this name.
		"""
		tmpfile, tmpname = tempfile.mkstemp()
		self.serializer_dict[name] = (tmpfile, tmpname)
		return tmpfile

	def write_out(self, scan_name):
		"""
		Move all temporary files to their permanent homes. Note that this clears the internal dictionary of filenames / contents.
		"""
		for name, (fd, fname) in self.serializer_dict.iteritems():
			self.serializer_dict[name] = numpy.loadtxt(fname)
		self.serializer_dict["segment"] = self.scan_segment
		self.serializer_dict["bandwidth"] = list(self.bandwidth) + [self.base_band]
		# FIXME: Reenable when it becomes available
		#numpy.savez_compressed(scan_name, **self.serializer_dict)
		numpy.savez(scan_name, **self.serializer_dict)
		self.serializer_dict.clear()

	def close(self):
		"""
		Close all temporary files.
		"""
		for (fd, fname) in self.serializer_dict.values():
			os.close(fd)


#
# =============================================================================
#
#                              Utility Functions
#
# =============================================================================
#

# Do this once per module load, since we may end up calling it a lot
__validattrs = [k for k, v in SnglBurst.__dict__.iteritems() if isinstance(v, types.MemberDescriptorType) or isinstance(v, types.GetSetDescriptorType)]
def convert_sngl_burst(snglburst, sb_table):
	"""
	Convert the snglburst object (presumed to be a pylal.xlal SnglBurst type) into lsctables version, as provided by the RowType() function of the supplied sb_table.
	"""
	event = sb_table.RowType()  # lsctables version
	for attr in __validattrs:
		# FIXME: This is probably slow
		setattr(event, attr, getattr(snglburst, attr))
	return event

############ From pipeline code
import os
import sys
import time
import signal
import glob
import tempfile
import threading
import math

from optparse import OptionParser, OptionGroup
import ConfigParser
from ConfigParser import SafeConfigParser

import numpy

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")

import gst

from gstlal import pipeparts
from gstlal.reference_psd import write_psd, read_psd_xmldoc

import gstlal.excesspower as ep
from gstlal import datasource

try:
    from glue import datafind
except ImportError:
    # FIXME: Remove when glue is totally updated to datafind
    from glue import GWDataFindClient as datafind

from glue.ligolw import ligolw, array, param, lsctables, table, ilwd
array.use_in(ligolw.LIGOLWContentHandler)
param.use_in(ligolw.LIGOLWContentHandler)
lsctables.use_in(ligolw.LIGOLWContentHandler)
from glue.ligolw import utils

from glue.segments import segment, segmentlist, segmentlistdict, PosInfinity
from glue import segmentsUtils
from glue import gpstime
from glue.lal import LIGOTimeGPS, Cache, CacheEntry

from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries

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
    # FIXME: Reincorpate these kwargs
    #epoch = laltypes.LIGOTimeGPS(0, message.structure["timestamp"]),
    #sampleUnits = laltypes.LALUnit(message.structure["sample-units"].strip()),
    new_psd = REAL8FrequencySeries(name = "PSD", f0 = 0.0, deltaF = elem.get_property("delta-f"), data = numpy.array(elem.get_property("mean-psd")))
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

#
# =============================================================================
#
#                             Options Handling
#
# =============================================================================
#

def append_options(parser=None):
    if parser is None:
        parser = OptionParser()

    parser.add_option("-f", "--initialization-file", dest="infile", help="Options to be pased to the pipeline handler. Required.", default=None)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Be verbose.", default=False)
    parser.add_option("-S", "--stream-tfmap", dest="stream_tfmap", action="store", help="Encode the time frequency map to video as it is analyzed. If the argument to this option is \"video\" then the pipeline will attempt to stream to a video source. If the option is instead a filename, the video will be sent to that name. Prepending \"keyframe=\" to the filename will start a new video every time a keyframe is hit.")
    parser.add_option("-r", "--sample-rate", dest="sample_rate", action="store", type="int", help="Sample rate of the incoming data.")
    parser.add_option("-t", "--disable-triggers", dest="disable_triggers", action="store_true", help="Don't record triggers.", default=False)
    parser.add_option("-T", "--file-cache", dest="file_cache", action="store_true", help="Create file caches of output files. If no corresponding --file-cache-name option is provided, an approrpriate name is constructed by default, and will only be created at the successful conclusion of the pipeline run.", default=False)
    parser.add_option("-F", "--file-cache-name", dest="file_cache_name", action="store", help="Name of the trigger file cache to be written to. If this option is specified, then when each trigger file is written, this file will be written to with the corresponding cache entry. See --file-cache for other details.", default=None)
    parser.add_option("-C", "--clustering", dest="clustering", action="store_true", default=False, help="Employ trigger tile clustering before output stage. Default or if not specificed is off." )
    parser.add_option("-m", "--enable-channel-monitoring", dest="channel_monitoring", action="store_true", default=False, help="Emable monitoring of channel statistics like even rate/signifiance and PSD power" )
    parser.add_option("-p", "--peak-over-sample-fraction", type=float, default=None, dest="peak_fraction", help="Take the peak over samples corresponding to this fraction of the DOF for a given tile. Default is no peak." )
    parser.add_option("-o", "--frequency-overlap", type=float, default=0.0, dest="frequency_overlap", help="Overlap frequency bands by this percentage. Default is 0." )
    parser.add_option("-d", "--drop-start-time", type=float, default=120.0, dest="drop_time", help="Drop this amount of time (in seconds) in the beginning of a run. This is to allow time for the whitener to settle to the mean PSD. Default is 120 s.")

    scan_sec = OptionGroup(parser, "Time-frequency scan", "Use these options to scan over a given segment of time for multiple time-frequency maps. Both must be specified to scan a segment of time. If the segment of time begins in the whitening segment, it will be clipped to be outside of it.")
    scan_sec.add_option("--scan-segment-start", type=float, dest="scan_start", help="Beginning of segment to scan.")
    scan_sec.add_option("--scan-segment-end", type=float, dest="scan_end", help="End of segment to scan.")
    parser.add_option_group(scan_sec)
    return parser


def process_options(options, gw_data_source_opts, pipeline, mainloop):
    # Locate and load the initialization file
    if not options.infile:
        print >>sys.stderr, "Initialization file required."
    elif not os.path.exists( options.infile ):
        print >>sys.stderr, "Initialization file path is invalid."
        sys.exit(-1)

    cfg = SafeConfigParser()
    cfg.read(options.infile)

    #
    # This supplants the ligo_data_find step and is mostly convenience
    #
    # TODO: Move to a utility library

    if gw_data_source_opts.data_source == "frames" and gw_data_source_opts.frame_cache is None:
        if gw_data_source_opts.seg is None:
            sys.exit("No frame cache present, and no GPS times set. Cannot query for data without an interval to query in.")

        # Shamelessly stolen from gw_data_find
        print "Querying LDR server for data location." 
        try:
            server, port = os.environ["LIGO_DATAFIND_SERVER"].split(":")
        except ValueError:
            sys.exit("Invalid LIGO_DATAFIND_SERVER environment variable set")
        print "Server is %s:%s" % (server, port)

        try:
            frame_type = cfg.get("instrument", "frame_type")
        except ConfigParser.NoOptionError:
            sys.exit("Invalid cache location, and no frame type set, so I can't query LDR for the file locations.")
        if frame_type == "":
            sys.exit("No frame type set, aborting.")

        print "Frame type is %s" % frame_type
        connection = datafind.GWDataFindHTTPConnection(host=server, port=port)
        print "Equivalent command line is "
        # FIXME: Multiple instruments?
        inst = gw_data_source_opts.channel_dict.keys()[0]
        print "gw_data_find -o %s -s %d -e %d -u file -t %s" % (inst[0], gw_data_source_opts.seg[0], gw_data_source_opts.seg[1], frame_type)
        cache = connection.find_frame_urls(inst[0], frame_type, gw_data_source_opts.seg[0], gw_data_source_opts.seg[1], urltype="file", on_gaps="error")

        tmpfile, tmpname = tempfile.mkstemp()
        print "Writing cache of %d files to %s" % (len(cache), tmpname)
        with open(tmpname, "w") as tmpfile:
        	cache.tofile(tmpfile)
        connection.close()
        gw_data_source_opts.frame_cache = tmpname

    handler = EPHandler(mainloop, pipeline)

    # Enable the periodic output of trigger statistics
    if options.channel_monitoring:
        handler.channel_monitoring = True

    # If a sample rate other than the native rate is requested, we'll need to 
    # keep track of it
    if options.sample_rate is not None:
        handler.rate = options.sample_rate

    # Does the user want a cache file to track the trigger files we spit out?
    if not options.file_cache:
        handler.output_cache = None

    # And if so, if you give us a name, we'll update it every time we output,
    # else only at the end of the run
    if options.file_cache and options.file_cache_name is not None:
        handler.output_cache_name = options.file_cache_name

    # Clustering on/off
    handler.clustering = options.clustering
    # Be verbose?
    handler.verbose = options.verbose

    # Instruments and channels
    # FIXME: Multiple instruments
    if len(gw_data_source_opts.channel_dict.keys()) == 1:
        handler.inst = gw_data_source_opts.channel_dict.keys()[0]
    else:
        sys.exit("Unable to determine instrument.")

    # FIXME: Multiple instruments
    if gw_data_source_opts.channel_dict[handler.inst] is not None:
        handler.channel = gw_data_source_opts.channel_dict[handler.inst]
    else:
        # TODO: In the future, we may request multiple channels for the same 
        # instrument -- e.g. from a single raw frame
        sys.exit("Unable to determine channel.")
    print "Channel name(s): " + handler.channel 

    # FFT and time-frequency parameters
    # Low frequency cut off -- filter bank begins here
    handler.flow = cfg.getfloat("tf_parameters", "min-frequency")
    # High frequency cut off -- filter bank ends here
    handler.fhigh = cfg.getfloat("tf_parameters", "max-frequency")
    # Frequency resolution of the finest filters
    handler.base_band = cfg.getfloat("tf_parameters", "min-bandwidth")
    # Tile duration should not exceed this value
    handler.max_duration = cfg.getfloat("tf_parameters", "max-duration")
    # Number of resolutions levels. Can't be less than 1, and can't be greater
    # than log_2((fhigh-flow)/base_band)
    handler.max_bandwidth = cfg.getfloat("tf_parameters", "max-bandwidth")
    handler.max_level = int(round(math.log(handler.max_bandwidth / handler.base_band, 2)))
    # Frequency band overlap -- in our case, handler uses 1 - frequency overlap
    if options.frequency_overlap > 1 or options.frequency_overlap < 0:
        sys.exit("Frequency overlap must be between 0 and 1.")
    handler.frequency_overlap = options.frequency_overlap

    # DOF options -- this affects which tile types will be calculated
    if cfg.has_option("tf_parameters", "max-dof"):
        handler.max_dof = cfg.getint("tf_parameters", "max-dof")
    if cfg.has_option("tf_parameters", "fix-dof"):
        handler.fix_dof = cfg.getint("tf_parameters", "fix-dof")

    if cfg.has_option("tf_parameters", "fft-length"):
        handler.fft_length = cfg.getfloat("tf_parameters", "fft-length")

    if cfg.has_option("cache", "cache-psd-every"):
        handler.cache_psd = cfg.getint("cache", "cache-psd-every")
        print "PSD caching enabled. PSD will be recorded every %d seconds" % handler.cache_psd
    else:
        handler.cache_psd = None

    if cfg.has_option("cache", "cache-psd-dir"):
        handler.cache_psd_dir = cfg.get("cache", "cache-psd-dir")
        print "Caching PSD to %s" % handler.cache_psd_dir
        
    # Used to keep track if we need to lock the PSD into the whitener
    psdfile = None
    if cfg.has_option("cache", "reference-psd"):
        psdfile = cfg.get("cache", "reference-psd")
        try:
            handler.psd = read_psd_xmldoc(utils.load_filename(psdfile, contenthandler = ligolw.LIGOLWContentHandler))[handler.inst]
            print "Reference PSD for instrument %s from file %s loaded" % (handler.inst, psdfile)
            # Reference PSD disables caching (since we already have it)
            handler.cache_psd = None
            handler.psd_mode = 1
        except KeyError: # Make sure we have a PSD for this instrument
            sys.exit( "PSD for instrument %s requested, but not found in file %s. Available instruments are %s" % (handler.inst, psdfile, str(handler.psd.keys())) )

    # Triggering options
    if cfg.has_option("triggering", "output-file-stride"):
        handler.dump_frequency = cfg.getint("triggering", "output-file-stride")
    if cfg.has_option("triggering", "output-directory"):
        handler.outdir = cfg.get("triggering", "output-directory")
    if cfg.has_option("triggering", "output-dir-format"):
        handler.outdirfmt = cfg.get("triggering", "output-dir-format")

    handler.output = not options.disable_triggers

    # FAP thresh overrides SNR thresh, because multiple resolutions will have 
    # different SNR thresholds, nominally.
    if cfg.has_option("triggering", "snr-thresh"):
        handler.snr_thresh = cfg.getfloat("triggering", "snr-thresh")
    if cfg.has_option("triggering", "fap-thresh"):
        handler.fap = cfg.getfloat("triggering", "fap-thresh")

    if handler.fap is not None:
        print "False alarm probability threshold (in Gaussian noise) is %g" % handler.fap
    if handler.snr_thresh is not None:
        print "Trigger SNR threshold sqrt(E/ndof-1) is %f" % handler.snr_thresh

    # Maximum number of events (+/- a few in the buffer) before which we drop an
    # output file
    if cfg.has_option("triggering", "events_per_file"):
        handler.max_events = cfg.get_int("triggering", "events_per_file")

    return handler

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

    postfirtee = pipeparts.mktee(pipeline, head)

    # Scan piece: Save the filtered time series
    if handler.trigger_segment:
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
        if min_rate > ep.EXCESSPOWER_UNIT_SCALE['mHz']:
            unit = 'mHz'
        elif min_rate > ep.EXCESSPOWER_UNIT_SCALE['uHz']:
            unit = 'uHz'
        elif min_rate > ep.EXCESSPOWER_UNIT_SCALE['nHz']:
            unit = 'nHz'
        else:
            sys.exit( "Requested undersampling rate would fall below 1 nHz." )
        # FIXME: No love for positive power of 10 units?

        handler.units = ep.EXCESSPOWER_UNIT_SCALE[unit]
        if handler.units != ep.EXCESSPOWER_UNIT_SCALE["Hz"]:
            head = pipeparts.mkgeneric(pipeline, head, "audioratefaker")
            head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float,rate=%d,channels=%d,width=64" % (handler.rate/handler.units, nchannels))

    # First branch -- send fully sampled data to wider channels for processing
    nlevels = int(numpy.ceil(numpy.log2(nchannels))) 
    for res_level in range(0, min(handler.max_level, nlevels)):
        # New level bandwidth
        band = handler.base_band * 2**res_level

        # The undersample_rate for band = R/2 is => sample_rate (passthrough)
        orig_rate = 2 * handler.base_band
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

        if max_samp < 2 and res_level == 0:
            sys.exit("The duration for the largest tile is smaller than a two degree of freedom tile. Try increasing the requested maximum tile duration or maximum DOF requirement.")
        elif max_samp < 2 and res_level != 0:
            print "Further resolution levels would result in tiles for which the maximum duration (%f) would not have enough DOF (2). Skipping remaining levels." % handler.max_duration
            break
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
                scan_obj.add_data_sink(pipeline, head, "sq_sum_series_level_%d_dof_1" % res_level, "time")

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
                scan_obj.add_data_sink(pipeline, durtee, "sq_sum_series_level_%d_dof_%d" % (res_level, ndof), "time")

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

            # Trigger generator
            # Number of tiles to peak over, if necessary
            peak_samples = max(1, int((peak_fraction or 0) * ndof))

            # Determine the SNR threshold for this trigger generator
            snr_thresh = ep.determine_thresh_from_fap(handler.fap, ndof)**2
            if verbose:
                print "SNR threshold for level %d, ndof %d: %f, will take peak over %d samples for this branch." % (res_level, ndof, snr_thresh, peak_samples)
            durtee = pipeparts.mkbursttriggergen(pipeline, durtee, n=peak_samples, bank_filename=handler.build_filter_xml(res_level, ndof, verbose=verbose), snr_thresh=snr_thresh)

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
