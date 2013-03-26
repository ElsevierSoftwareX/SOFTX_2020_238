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
from StringIO import StringIO

import numpy

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst

from gstlal.pipeutil import gst
from gstlal.simplehandler import Handler
from gstlal.reference_psd import write_psd, read_psd_xmldoc

import gstlal.excesspower as ep
from gstlal.inspiral import add_cbc_metadata

from glue.ligolw import ligolw
from glue.ligolw import array
from glue.ligolw import param
from glue.ligolw import lsctables
from glue.ligolw import table

class ContentHandler(ligolw.LIGOLWContentHandler): 
	pass 
array.use_in(ligolw.LIGOLWContentHandler)
param.use_in(ligolw.LIGOLWContentHandler)
lsctables.use_in(ligolw.LIGOLWContentHandler)

from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments

from glue.segments import segment, segmentlist, segmentlistdict, PosInfinity
from glue import segmentsUtils
from glue import gpstime
from glue.lal import LIGOTimeGPS, Cache, CacheEntry

from pylal import snglcluster
from pylal import ligolw_bucluster
from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries
from pylal.xlal.datatypes.lalunit import LALUnit
#from pylal.xlal.datatypes.snglburst import SnglBurst

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
		# FIXME: This should be ~number of PSDs stored in history * fft_length
		# Right now that's probably something like 80 s.
		self.whitener_offset = 40

		# Keep track of units -- enable us to go below rates of 1 Hz
		self.units = ep.EXCESSPOWER_UNIT_SCALE['Hz']

		# Defaults -- Time-frequency map settings
		self.base_band = 16
		self.flow = 64 
		self.fhigh = 1000
		self.fft_length = 8 # s

		# Defaults -- Resolution settings
		self.rate = 2048
		#self.rate = 4096
		self.max_level = 1

		# Defaults -- filtering
		self.filter_len = 2*int(2*self.rate/self.base_band)
		self.filter_xml = {}
		self.filter_bank = None
		self.freq_filter_bank = None
		# TODO: Maybe not necessary
		self.firbank = None

		# Defaults -- PSD settings
		self.whitener = None
		# Current measured PSD from the whitener
		self.prev_psd = self.psd = None
		# This is used to store the previous value of the PSD power
		self.psd_power = 0
		self.cache_psd = None
		self.cache_psd_dir = "./"
		self.last_psd_cache = 0
		self.psd_change_thresh = 0.2
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
		self.outfile = "test.xml"
		self.outdir = "./"
		self.outdirfmt = ""
		self.make_output_table()
		self.output_cache = Cache()
		self.output_cache_name = None
		self.snr_thresh = 5.5
		self.fap = None
		self.dump_frequency = 600 # s
		self.max_events = 1e6
		self.time_since_dump = self.start
		self.db_thresh = None  # off
		self.db_client = None  # off

		self.trigger_segment = None

		self.spec_corr = self.build_default_correlation( self.rate )

		# required for file locking
		self.lock = threading.Lock()

		self.bus = pipeline.get_bus()
		self.bus.add_signal_watch()
		self.bus.connect( "message", self.process_message )

		self.pipeline = pipeline

		self.verbose = False
		self.clustering = False
		self.channel_monitoring = False
		self.stats = ep.SBStats()
		self.filter_rebuild_times = []

		super(type(self), self).__init__(mainloop, pipeline)

	def set_trigger_time_and_action( self, trig_seg, action=["psd"] ):
		"""
		Inform the handler of a specific time of interest.
		"""
		# TODO: Bounds checking
		self.trigger_segment = trig_seg

		# TODO: Handle only specific action requests

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
			print >>sys.stderr, "Got latency message, ignoring for now."
			return
		elif message.structure is None: 
			print >>sys.stderr, "Got message with type: %s ...but no handling logic, so ignored." % str(message.type)
			return

		# TODO: Move this to PSD difference checker
		if message.structure.get_name() == "spectrum":
			# FIXME: Units
			ts = message.structure[ "timestamp" ]*1e-9
			self.filter_rebuild_times.append( ("spectrum_message", ts) )
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

		# FIXME: This isn't set properly only on the first try --- that is to say,
		# the value of the latency for the firbank is reset, but the setting has np
		# further effect on any of the timestamps emitted by the FIR bank. This is
		# notedi, but not resolved in gstlal_firbank.c
		firbank.set_property( "latency", len(firbank.get_property("fir_matrix")[0])/2 )
		self.firbank = firbank

	def add_matmixer( self, mm, res_level ):
		self.mmixers[ res_level ] = mm
		self.rebuild_matrix_mixers( res_level )

	def build_default_psd( self, rate, df, fhigh ):
		"""
		Builds a dummy PSD to use until we get the right one.
		"""
		psd = REAL8FrequencySeries()
		psd.deltaF = df
		psd.sampleUnits = LALUnit( "s strain^2" )
		psd.data = numpy.ones( int(rate/2/df) + 1 ) 
		psd.f0 = 0
		self.psd = psd
		return psd

	def build_default_correlation( self, rate ):
		"""
		Builds a Kronecker delta correlation series for k, k'.
		"""
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
				band = self.base_band
			)
			cmatrix = ep.build_chan_matrix( 
				nchannels = nchannels,
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
		self.filter_bank, self.freq_filter_bank = ep.build_filter( 
			fhigh = self.fhigh, 
			flow=self.flow, 
			rate=self.rate, 
			psd = self.psd, 
			corr = self.spec_corr, 
			b_wind = self.base_band 
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
			self.inst
		)
		output = "%sgstlal_excesspower_bank_%s_%s_level_%d_%d.xml" % (loc, self.inst, self.channel, res_level, ndof)
		self.filter_xml[output] = self.filter_xml[(res_level, ndof)]
		utils.write_filename( self.filter_xml[output], output, verbose = verbose,
		       gz = (output or "stdout").endswith(".gz") )
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
			max_level = self.max_level
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
		print >>sys.stderr, "New filter latency %d (%f s)" % (latency, latency/float(self.rate))

		if self.verbose:
			print >>sys.stderr, "Rebuilding matrix mixer"
		#self.rebuild_chan_mix_matrix()
		# Rebuild the matrix mixer with new normalization coefficients
		self.rebuild_matrix_mixers()

	def make_process_tables( self, options=None, xmldoc=None ):
		"""
		Create a process and process_params table for use in the output document. If the options parameter is passed, the process_params table will be created from it. If the output xmldoc is provided, it will register the pipeline with the document. Returns the process and process_params table, however, note that the process table will be empty if no options are provided.
		"""
		if options:
			self.process_params = vars( options )

		if self.process_params is None:
			print >>sys.stderr, "WARNING: Options have not yet been set in the handler. Process and ProcessParams may not be constructed properly. Call handler.make_process_tables() with the options argument to set command line options."
			return lsctables.New(lsctables.ProcessTable)

		if xmldoc and self.process is None:
			self.process = ligolw_process.register_to_xmldoc( xmldoc, "gstlal_excesspower", self.process_params )
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

	# TODO: Move this into library code
	def make_output_table( self ):
		self.triggers = lsctables.New(lsctables.SnglBurstTable,
			["ifo", "peak_time", "peak_time_ns", "start_time", "start_time_ns",
			"duration",  "search", "event_id", "process_id",
			"central_freq", "channel", "amplitude", "snr", "confidence",
			"chisq", "chisq_dof", "bandwidth"])
			#"peak_frequency",
			#"stop_time", "peak_time_ns", "start_time_ns", "stop_time_ns",
 			#"time_lag", "flow", "fhigh", tfvolume, hrss, process_id
		return self.triggers

	def process_triggers( self, newtrigs, cluster_passes=0 ):
		"""
		Add additional information to triggers from the buffer and cluster them with the triggers already present in the handler. The cluster parameter controls how many passes of the clustering routine should be performed, with True being a special value indicating "as many as neccessary".
		"""

		output = ligolw.Document()
		output.appendChild(ligolw.LIGO_LW())
		# FIXME: this is probably broken since it's going to create multiple
		# process ids for one run of excesspower since the program gets registered
		# every time this function is called, which is often.
		process = self.make_process_tables( None, output )

		# Assign process ids to events
		for trig in newtrigs:
			# For some reason, importing the XLAL SnglBurst 
			# causes the write to crash -- we still need a better
			# way to identified processed triggers
			if trig.process_id is not None:
				continue
			trig.process_id = self.process.process_id
			# If we're using a different units system, adjust back to SI
			trig.duration *= self.units
			#trig.peak_time /= self.units
			# Readjust start time for units
			trig.start_time -= self.start
			trig.start_time /= self.units
			trig.start_time += self.start

		# FIXME: Write only the triggers that aren't XML
		if len(newtrigs) != 0:
			# Just in case any of the handler's trigger tables are the XLAL snglburst
			output.childNodes[0].appendChild( newtrigs )

			# Do a temporary write to make the SnglBurst objects into XML rows -- 
			# this should probably be done only if clustering is requested, since 
			# it's the only thing that thinks the triggers should be XML rows

			self.lock.acquire()
			# Enable to debug LIGOLW stream
			#utils.write_fileobj(output, sys.stdout)
			# TODO: Find out if this is any faster
			tmpfile = StringIO()
			utils.write_fileobj(output, tmpfile, trap_signals = None)
			self.lock.release()

			# Reload the document to convert the SnglBurst type to rows
			tmpfile.seek(0)
			output, mdhash = utils.load_fileobj(tmpfile, contenthandler=ContentHandler)
			ligolw_bucluster.add_ms_columns( output )
			tmpfile.close()

			newtrigs = table.get_table( output, lsctables.SnglBurstTable.tableName ) 
		self.triggers.extend( newtrigs )

		# Avoid all the temporary writing and the such
		if cluster_passes == 0 or not self.clustering:
			return

		full = cluster_passes is True
		if full:
			cluster_passes = 1
		# Pipe down unless its important
		verbose = self.verbose and full
		changed = True
		off = ligolw_bucluster.ExcessPowerPreFunc( self.triggers )
		while changed and self.clustering:
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
		ligolw_bucluster.ExcessPowerPostFunc( self.triggers, off )
		#self.triggers = newtrigs

	def write_triggers( self, flush=True, filename=None, output_type="xml" ):

		if not self.output: 
			return
		if filename == None:
			filename = self.outfile

		output = ligolw.Document()
		output.appendChild(ligolw.LIGO_LW())

		requested_segment = segment(
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

		output.childNodes[0].appendChild( self.triggers )

		add_cbc_metadata( output, process, requested_segment )
		search_sum = lsctables.table.get_table( output, lsctables.SearchSummaryTable.tableName )
		search_sum[0].comment = ",".join( [ "%s:%10.9f" % (m[0], m[1]) for m in self.filter_rebuild_times ] )
		self.filter_rebuild_times = []
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
		if self.current_segment is not None and float(self.current_segment[0]) > self.analysis_segment[1]:
			# add the current segment
			cur_seg = segment( self.current_segment[0], LIGOTimeGPS(analysis_segment[1]) )
			self.seglist["state"].append( cur_seg )
			# TODO: send the new time to handle_segment instead
			self.current_segment = segment( cur_seg[1], PosInfinity )

		# Write segments
		llwseg = ligolw_segments.LigolwSegments( output )
		# FIXME: Better names and comments?
		llwseg.insert_from_segmentlistdict( self.seglist, u"gstlal_excesspower segments \u263b", comment="gstlal_excesspower segments" )

		llwseg.finalize(process)

		# FIXME: We should be careful to not fragment segments across output too
		# much
		self.seglist.clear()

		self.seglist["state"] = segmentlist([])

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
				print >>sys.stderr, "Event significance:\nSNR\t\tsig"
				for (snr, sig) in esig:
					print >>sys.stderr, "%4.2f\t\t%4.2f" % (snr, sig)

			self.lock.acquire()
			jsonf = open( "%s-%s-channel_mon.json" % (self.inst, self.channel), "w" )
			print >>jsonf, json.dumps( stat_json )
			jsonf.close()
			self.lock.release()

		if flush: 
			self.make_output_table()
		if self.db_thresh is None: 
			return

		# FIXME: get_table doesn't return the type of table you want it just 
		# returns a "Table" this is probbably why the upload_tbl needs the full
		# definition
		clus_triggers = table.get_table( output, lsctables.SnglBurstTable.tableName )

		for sb in filter( lambda sb : sb.snr > self.db_thresh**2, clus_triggers ):
			# TODO: Merge these two
			#if sb.peak_time in self.discard_segment: continue
			if sb.peak_time not in analysis_segment: continue
			upload_tbl = lsctables.New(lsctables.SnglBurstTable,
			["ifo", "peak_time", "peak_time_ns", "start_time", "start_time_ns",
			"duration",  "search", "event_id", "process_id",
			"central_freq", "channel", "amplitude", "snr", "confidence",
			"chisq", "chisq_dof", "bandwidth"])

			upload_tbl.append( sb )
			ep.upload_to_db( upload_tbl, search = "EP", db = self.db_client )
		
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

		subdir = ""
		if self.outdirfmt is not None:
			subdir = ep.append_formatted_output_path( self.outdirfmt, self )
		outfile = ep.make_cache_parseable_name(
			inst = self.inst,	
			tag = self.channel,
			start = self.time_since_dump,
			stop = self.stop,
			ext = "xml",
			dir = self.outdir + subdir
		)
		self.write_triggers( False, filename = outfile )

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

