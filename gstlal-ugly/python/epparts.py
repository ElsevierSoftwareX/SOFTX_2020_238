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

from pylal import ligolw_bucluster
from pylal.xlal.datatypes.real8frequencyseries import REAL8FrequencySeries

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
		self.filter_bank = None
		# TODO: Maybe not necessary
		self.firbank = None

		# Defaults -- PSD settings
		self.psd = None
		# This is used to store the previous value of the PSD power
		self.psd_power = 0
		self.cache_psd = None
		self.cache_psd_dir = "./"
		self.last_psd_cache = 0
		self.psd_change_thresh = 0.5 # fifty percent

		# Defaults -- Two-point spectral correlation settings
		self.cache_spec_corr = False

		# Defaults -- mixer matrices
		self.chan_matrix = None
		self.mmixers = {}

		# Defaults -- data products
		self.output = True
		self.triggers = None
		self.process_params = None
		self.process = None
		self.outfile = "test.xml"
		self.outdir = "./"
		self.make_output_table()
		self.output_cache = Cache()
		self.output_cache_name = None
		self.snr_thresh = 5.5
		self.fap = None
		self.dump_frequency = 600 # s
		self.max_events = 1e4
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
			if self.verbose:
				print >>sys.stderr, "Ending segment #%d: %s" % (len(self.seglist["state"]), str(self.seglist["state"][-1]))
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
			print >>sys.stderr, "Got message with type: %s ...but no handling logic, so ignored."
			return

		# TODO: Move this to PSD difference checker
		if message.structure.get_name() == "spectrum":
			# FIXME: Units
			ts = message.structure[ "timestamp" ]*1e-9
			if self.trigger_segment is not None and ts in self.trigger_segment:
				self.dump_psd( timestamp = ts )
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
			dir = "./"
		)

		write_psd( filename, { self.inst: self.psd } )

	def add_firbank( self, firbank ):
		"""
		Set the main base band FIR bank, and build the FIR matrix.
		"""
		self.firbank = firbank
		firbank.set_property( "fir-matrix", self.rebuild_filter() )

	def add_matmixer( self, mm, res_level ):
		self.mmixers[ res_level ] = mm
		self.rebuild_matrix_mixers( res_level )

	def build_default_psd( self, rate, filter_len ):
		"""
		Builds a dummy PSD to use until we get the right one.
		"""
		psd = REAL8FrequencySeries()
		psd.deltaF = float(rate)/filter_len
		psd.data = numpy.ones( filter_len/2 + 1 ) 
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
			if res_level != None and res_level != i: 
				continue

			nchannels = self.filter_bank.shape[0]
			up_factor = int(numpy.log2(nchannels/(nchannels >> i)))
			cmatrix = ep.build_chan_matrix( 
				nchannels = nchannels,
				up_factor = up_factor,
				norm = self.chan_matrix[i] 
			)
			mm.set_property( "matrix", cmatrix )

	def rebuild_filter( self ):
		"""
		Calling this function rebuilds the filter FIR banks and assigns them to their proper element. This is normally called when the PSD or spectrum correlation changes.
		"""
		self.filter_bank = ep.build_filter( 
			fhigh = self.fhigh, 
			flow=self.flow, 
			rate=self.rate, 
			psd = self.psd, 
			corr = self.spec_corr, 
			b_wind = self.base_band 
		)
		return self.filter_bank

	def build_filter_xml( self, res_level, loc="" ):
		"""
		Calls the EP library to create a XML of sngl_burst tables representing the filter banks. At the moment, this dumps them to the current directory, but this can be changed by supplying the 'loc' argument. The written filename is returned for easy use by the trigger generator.
		"""
		self.filter_xml = ep.create_bank_xml(
			self.flow,
			self.fhigh,
			self.base_band*(res_level+1),
			# FIXME: Is there a factor of two here? -- No, remove the factor of two...
			1.0 / (2*self.base_band*(res_level+1)), # resolution level starts from 0
			self.inst
		)
		output = "%sgstlal_excesspower_bank_%s_%s_level_%d.xml" % (loc, self.inst, self.channel, res_level)
		utils.write_filename( self.filter_xml, output, verbose = True,
		       gz = (output or "stdout").endswith(".gz") )
		return output

	def destroy_filter_xml( self, loc="" ):
		"""
		FIXME: This is really inelegant.
		"""
		for f in glob.glob( "%sgstlal_excesspower_bank_%s_%s_level_*.xml" % (loc, self.inst, self.channel) ):
			os.remove( f )

	def rebuild_chan_mix_matrix( self ):
		"""
		Calling this function rebuilds the matrix mixer coefficients for higher resolution components. This is normally called when the PSD or spectrum correlation changes.
		"""
		self.chan_matrix = ep.build_inner_product_norm( 
			corr = self.spec_corr, 
			band = self.base_band, 
			del_f = self.psd.deltaF,
			nfilts = len(self.filter_bank),
			flow = self.flow,
			# TODO: PSD option to lalburst IP doesn't work
			# This should be fixed, let's reenable it
			#psd = self.psd
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

		if self.verbose:
			print >>sys.stderr, "Rebuilding matrix mixer"
		self.rebuild_chan_mix_matrix()
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

	# TODO: Move this into library code
	def write_triggers( self, flush=True, overwrite=False, filename=None ):
		if not self.output: 
			return
		if len(self.triggers) == 0: 
			return

		if filename == None:
			filename = self.outfile

		output = ligolw.Document()
		output.appendChild(ligolw.LIGO_LW())

		requested_segment = segment(
			LIGOTimeGPS( self.time_since_dump ), 
			LIGOTimeGPS( self.stop )
		)

		analysis_segment = requested_segment
		# If we include start up time, indicate it in the search summary
		self.whiten_seg = segment( 
			LIGOTimeGPS(self.start), 
			LIGOTimeGPS(self.start + self.whitener_offset)
		)
		if self.whiten_seg.intersects( analysis_segment ):
			if analysis_segment in self.whiten_seg:
				# All the analyzed time is within the settling time
				# We make this explicit because the segment constructor will just reverse the arguments if arg2 < arg1 and create an incorrect segment
				analysis_segment = segment( 
					analysis_segment[1], analysis_segment[1]
				)
			else:
				analysis_segment -= self.whiten_seg

		process = self.make_process_tables( None, output )
		"""
		process_params = vars( options )
		process = ligolw_process.register_to_xmldoc( output, "gstlal_excesspower", vars(options) )
		process.set_ifos( [self.inst] )
		"""

		# Assign process ids to events
		for trig in self.triggers:
			trig.process_id = process.process_id
			# If we're using a different units system, adjust back to SI
			trig.duration *= self.units
			#trig.peak_time /= self.units
			# Readjust start time for units
			trig.start_time -= self.start
			trig.start_time /= self.units
			trig.start_time += self.start

		output.childNodes[0].appendChild( self.triggers )

		add_cbc_metadata( output, process, requested_segment )
		search_sum = lsctables.table.get_table( output, lsctables.SearchSummaryTable.tableName )
		# TODO: This shouldn't set every one of them in case we reuse XML 
		# documents later
		for row in search_sum:
			row.set_out( analysis_segment )

		if self.current_segment is not None:
			# add the current segment
			cur_seg = segment( self.current_segment[0], LIGOTimeGPS(analysis_segment[1]) )
			self.seglist["state"].append( cur_seg )

		# Write segments
		llwseg = ligolw_segments.LigolwSegments( output )
		# FIXME: Better names and comments?
		llwseg.insert_from_segmentlistdict( self.seglist, "gstlal_excesspower segments", comment="gstlal_excesspower segments", version=u'\u263b' )

		llwseg.finalize(process)

		# FIXME: We should be careful to not fragment segments across output too
		# much
		self.seglist.clear()

		# FIXME: We actually will end up writing the same segment more than once
		# across adjacent files. This is probably okay but, fair warning.
		# What maybe should be done is to intersect the final segment with the
		# analysis_segment's end and then make the current segment have the same
		# start time as the last written segment's end.
		self.seglist["state"] = segmentlist([])

		# Do a temporary write to make the SnglBurst objects into XML rows -- 
		# this should probably be done only if clustering is requested, since 
		# it's the only thing that thinks the triggers should be XML rows

		# TODO: replace cbc filter table with our own
		#cbc_filter_table = lsctables.getTablesByType( output, lsctables.FilterTable )[0]
		#ep_filter_table = lsctables.getTablesByType( self.filter_xml, lsctables.FilterTable )[0]
		#output.replaceChild( ep_filter_table, cbc_filter_table )
		print >>sys.stderr, "Outputting triggers for %s\n" % str(requested_segment)

		# write the new distribution stats to disk
		self.lock.acquire()
		# Enable to debug LIGOLW stream
		#utils.write_fileobj(output, sys.stdout)
		# TODO: Get a temporary file name to write to -- once we do that we can
		# handle max events much easier, because the temporary file can be 
		# deleted and we only write if clustering is on and events > max
		utils.write_filename(output, filename, verbose = self.verbose, gz = (filename or "stdout").endswith(".gz"), trap_signals = None)
		self.lock.release()

		# Reload the document to convert the SnglBurst type to rows
		output = utils.load_filename(filename, verbose = self.verbose)
		process = lsctables.table.get_table( output, lsctables.ProcessTable.tableName )[0]

		# FIXME: Should this be moved to the trigger import function?
		changed = True

		# We need this because we might have just set it to be an 
		# infinitesimally small segment. Clustering will choke on this, and we 
		# want to know what we can send to dbs anyway.
		while changed and self.clustering and abs(analysis_segment) != 0:
			ligolw_bucluster.add_ms_columns( output )
			output, changed = ligolw_bucluster.ligolw_bucluster( 
				xmldoc = output,
				program = "gstlal_excesspower",
				process = process,
				prefunc = ligolw_bucluster.ExcessPowerPreFunc,
				postfunc = ligolw_bucluster.ExcessPowerPostFunc,
				testfunc = ligolw_bucluster.ExcessPowerTestFunc,
				clusterfunc = ligolw_bucluster.ExcessPowerClusterFunc,
				sortfunc = ligolw_bucluster.ExcessPowerSortFunc,
				bailoutfunc = ligolw_bucluster.ExcessPowerBailoutFunc,
				verbose = self.verbose
			)

		# TODO: Respect max events by removing output XML file and returning
		# if clustering reduces trigger number below required.
		if self.clustering:
			# write the new distribution stats to disk
			self.lock.acquire()
			# Enable to debug LIGOLW stream
			#utils.write_fileobj(output, sys.stdout)
			utils.write_filename(output, filename, verbose = self.verbose, gz = (filename or "stdout").endswith(".gz"), trap_signals = None)
			self.lock.release()

		# TODO: We ignore triggers in the whitening segment now anyway, let's
		# just pull this unless we have reason to keep it
		#self.discard_segment = segment( 
			#LIGOTimeGPS(self.start), 
			#LIGOTimeGPS(self.start + 300)
		#)

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
			self.output_cache.tofile( file(self.output_cache_name, "a") )
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
			upload_to_db( upload_tbl, search = "EP", db = self.db_client )
		
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

		outfile = ep.make_cache_parseable_name(
			inst = self.inst,	
			tag = self.channel,
			start = self.time_since_dump,
			stop = self.stop,
			ext = "xml",
			dir = self.outdir
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

