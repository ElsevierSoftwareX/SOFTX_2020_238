# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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

## 
# @file
#
# A file that contains the lloidparts module code; Roughly speaking it
# implements the algorithm described in <a
# href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>
#
#
# Review Status
#
# | Names                                 | Hash                                     | Date       | Diff to Head of Master      |
# | ------------------------------------- | ---------------------------------------- | ---------- | --------------------------- |
# | Sathya, Duncan Me, Jolien, Kipp, Chad | 2f5f73f15a1903dc7cc4383ef30a4187091797d1 | 2014-05-02 | <a href="@gstlal_inspiral_cgit_diff/python/lloidparts.py?id=HEAD&id2=2f5f73f15a1903dc7cc4383ef30a4187091797d1">lloidparts.py</a> |
#
# #### Actions
# - Feature request: do checkpointing when instruments are down
# - Feature request: need to hook up equivalent of "CAT 2" vetoes to online analysis when ready, and HW inj veto, etc.
# - Document the parameters in mkLLOIDmulti()
# - Check if bank ids are the same across different instruments
# - Feature request: Make time-frequency videos for interesting events
# - Inject signals of known, high SNR and measure loss
# - move the makesegmentsrcgate to before the matrix mixer, not in the current sum-of-squares control loop
# - Make conditional branches in the graph gray boxes
# - consider zero padding the beginning of jobs to get rid of mkdrop()
# - Decide how to properly normalize SNRs for incomplete filter segments (currently filters are not renormalized so the loss in SNR is bigger than might be expected)
# - Check and possibly fix the function that switches between time-domain and FFT convolution based on stride and number of samples
# - Consider if quality = 9 is right for downsampling (it should be plenty good, but maybe excessive?)
#
#
# #### Functions/classes not reviewed since they will be moved
# - DetectorData 
# - mkSPIIRmulti
# - mkSPIIRhoftToSnrSlices
# - mkLLOIDSnrSlicesToTimeSliceChisq
# - mkLLOIDSnrChisqToTriggers

##
# @package lloidparts
#
# a module for building gstreamer graphs of the LLOID algorithm
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import math
import numpy
import os
import StringIO
import sys
import warnings


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import process as ligolw_process
from gstlal import bottle
from gstlal import datasource
from gstlal import multirate_datasource
from gstlal import pipeio
from gstlal import pipeparts
from gstlal import simplehandler
import lal
from lal import LIGOTimeGPS


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#

##
# A "sum-of-squares" aggregator
# 
# _Gstreamer graph describing this function:_
#
# @dot
# digraph G {
#	rankdir="LR";
#
#	// nodes
#	node [shape=box, style=rounded];
#
#	lal_adder
#	lal_peak [URL="\ref pipeparts.mkpeak()", style=filled, color=grey, label="lal_peak\niff control_peak_samples > 0"];
#	capsfilter [URL = "\ref pipeparts.mkcapsfilter()"];
#	"mksegmentsrcgate()" [URL="\ref datasource.mksegmentsrcgate()"];
#	tee [URL="\ref pipeparts.mktee()"];
#	lal_checktimestamps [URL="\ref pipeparts.mkchecktimestamps()"];
#
#	// connections
#	"? sink 1" -> lal_adder;
#	"? sink 2" -> lal_adder;
#	"? sink N" -> lal_adder;
#	lal_adder -> capsfilter;
#	capsfilter -> lal_peak;
#	lal_peak -> "mksegmentsrcgate()";
#	"mksegmentsrcgate()" -> lal_checktimestamps;
#	lal_checktimestamps -> tee;
#	tee -> "? src";
# }
# @enddot
#
#
def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None, reconstruction_segment_list = None, control_peak_samples = None):
	"""!
	This function implements a portion of a gstreamer graph to provide a
	control signal for deciding when to reconstruct physical SNRS

	@param pipeline A reference to the gstreamer pipeline in which to add this graph
	@param rate An integer representing the target sample rate of the resulting src
	@param verbose Make verbose
	@param suffix Log name for verbosity
	@param reconstruction_segment_list A segment list object that describes when the control signal should be on.  This can be useful in e.g., only reconstructing physical SNRS around the time of injections, which can save an enormous amount of CPU time.
	@param control_peak_samples If nonzero, this would do peakfinding on the control signal with the window specified by this parameter.  The peak finding would give a single sample of "on" state at the peak.   This will cause far less CPU to be used if you only want to reconstruct SNR around the peak of the control signal. 
	"""
	#
	# start with an adder and caps filter to select a sample rate
	#

	snk = pipeparts.mkadder(pipeline, None)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw, rate=%d" % rate)

	#
	# Add a peak finder on the control signal sample number
	#

	if control_peak_samples > 0:
		src = pipeparts.mkpeak(pipeline, src, control_peak_samples)

	#
	# optionally add a segment src and gate to only reconstruct around
	# injections
	#
	# FIXME:  set the names of these gates so their segments can be
	# collected later?  or else propagate this segment list into the
	# output some other way.

	if reconstruction_segment_list is not None:
		src = datasource.mksegmentsrcgate(pipeline, src, reconstruction_segment_list, invert_output = False)

	#
	# verbosity and a tee
	#

	logname = suffix and "_%s" % suffix or ""
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_sumsquares%s" % logname)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps%s_sumsquares" % logname)
	src = pipeparts.mktee(pipeline, src)

	#
	# return the adder and tee
	#

	return snk, src


class Handler(simplehandler.Handler):
	"""!
	A subclass of simplehandler.Handler to be used with e.g.,
	gstlal_inspiral

	Implements additional message handling for dealing with spectrum
	messages and checkpoints for the online analysis including periodic
	dumps of segment information, trigger files and background
	distribution statistics.
	"""
	def __init__(self, mainloop, pipeline, dataclass, instruments, tag = "", seglistdict = None, zero_lag_ranking_stats_filename = None, segment_history_duration = LIGOTimeGPS(2592000), verbose = False):
		"""!
		@param mainloop The main application's event loop
		@param pipeline The gstreamer pipeline that is being controlled by this handler
		@param dataclass An inspiral.Data class instance
		@param tag The tag to use for naming file snapshots, e.g. the description will be "%s_LLOID" % tag
		@param verbose Be verbose
		"""
		super(Handler, self).__init__(mainloop, pipeline)

		self.dataclass = dataclass

		self.tag = tag
		self.zero_lag_ranking_stats_filename = zero_lag_ranking_stats_filename
		self.segment_history_duration = segment_history_duration
		self.verbose = verbose

		# setup segment list collection from gates
		#
		# FIXME:  knowledge of what gates are present in what
		# configurations, what they are called, etc., somehow needs to live
		# with the code that constructs the gates
		#
		# FIXME:  in the offline pipeline, state vector segments
		# don't get recorded.  however, except for the h(t) gate
		# segments these are all inputs to the pipeline so it
		# probably doesn't matter.  nevertheless, they maybe should
		# go into the event database for completeness of that
		# record, or maybe not because it could result in a lot of
		# duplication of on-disk data.  who knows.  think about it.
		gate_suffix = {
			# FIXME uncomment the framesegments line once the
			# online analysis has a frame segments gate
			#"framesegments": "frame_segments_gate",
			"statevectorsegments": "state_vector_gate",
			"whitehtsegments": "ht_gate"
		}

		# dictionary mapping segtype to segmentlist dictionary
		# mapping instrument to segment list
		self.seglistdicts = dict((segtype, segments.segmentlistdict((instrument, segments.segmentlist()) for instrument in instruments)) for segtype in gate_suffix)
		# add a "triggersegments" entry
		if seglistdict is None:
			self.seglistdicts["triggersegments"] = segments.segmentlistdict((instrument, segments.segmentlist()) for instrument in instruments)
		else:
			self.seglistdicts["triggersegments"] = seglistdict
		# hook the Data class's livetime record keeping into ours
		# so that all segments come here
		# FIXME:  don't do this, get rid of the Data class
		dataclass.seglistdicts = self.seglistdicts

		# create a deep copy to keep track of cumulative segments
		self.cumulative_seglistdicts = self.seglistdicts.copy()

		# state of segments being collected
		self.current_segment_start = {}

		# iterate over segment types and instruments, look for the
		# gate element that should provide those segments, and
		# connect handlers to collect the segments
		if verbose:
			print >>sys.stderr, "connecting segment handlers to gates ..."
		for segtype, seglistdict in self.seglistdicts.items():
			for instrument in seglistdict:
				try:
					name = "%s_%s" % (instrument, gate_suffix[segtype])
				except KeyError:
					# this segtype doesn't come from
					# gate elements
					continue
				elem = self.pipeline.get_by_name(name)
				if elem is None:
					# ignore missing gate elements
					if verbose:
						print >>sys.stderr, "\tcould not find %s for %s '%s'" % (name, instrument, segtype)
					continue
				if verbose:
					print >>sys.stderr, "\tfound %s for %s '%s'" % (name, instrument, segtype)
				elem.connect("start", self.gatehandler, (segtype, instrument, "on"))
				elem.connect("stop", self.gatehandler, (segtype, instrument, "off"))
				elem.set_property("emit-signals", True)
		if verbose:
			print >>sys.stderr, "... done connecting segment handlers to gates"

		# most recent spectrum reported by each whitener
		self.psds = {}
		bottle.route("/psds.xml")(self.web_get_psd_xml)

		# segment lists
		bottle.route("/segments.xml")(self.web_get_segments_xml)
		bottle.route("/cumulative_segments.xml")(self.web_get_cumulative_segments_xml)

	def do_on_message(self, bus, message):
		"""!
		Override the on_message method of simplehandler to handle
		additional message types, e.g., spectrum and checkpointing messages.

		@param bus A reference to the pipeline's bus
		@param message A reference to the incoming message
		"""
		if message.type == Gst.MessageType.ELEMENT:
			if message.get_structure().get_name() == "spectrum":
				# get the instrument, psd, and timestamp.
				# NOTE: epoch is used for the timestamp, this
				# is the middle of the most recent FFT interval
				# used to obtain this PSD
				instrument = message.src.get_name().split("_")[-1]
				psd = pipeio.parse_spectrum_message(message)
				timestamp = psd.epoch

				# save
				self.psds[instrument] = psd

				# update horizon distance history
				#
				# FIXME:  probably need to compute these for a
				# bunch of masses.  which ones?
				self.dataclass.record_horizon_distance(instrument, timestamp, psd, m1 = 1.4, m2 = 1.4)
				return True
		elif message.type == Gst.MessageType.APPLICATION:
			if message.get_structure().get_name() == "CHECKPOINT":
				# FIXME make a custom parser for CHECKPOINT messages?
				timestamp = message.get_structure()["timestamp"]
				# FIXME:  the function that makes these
				# messages uses a default value of None,
				# and in principle could be called with
				# other non-timestamp-like things, so this
				# code should check that it has actually
				# gotten a valid timestamp
				self.checkpoint(timestamp)
				return True
		elif message.type == Gst.MessageType.EOS:
			with self.dataclass.lock:
				# FIXME:  how to choose correct timestamp?
				try:
					timestamp = self.seglistdicts["triggersegments"].extent_all()[1].ns()
				except ValueError:
					# no segments
					return False
			self.close_segments(timestamp)
			return False
		return False

	def _close_segments(self, timestamp):
		"""
		@timestamp must be a GPS time that is guaranteed to precede
		any possible future state transitions in all segment lists
		being tracked.
		"""
		# close out existing segments.  the code in the loop
		# modifies the iteration target, so iterate over a copy
		for (segtype, instrument), start_time in list(self.current_segment_start.items()):
			if timestamp < start_time.ns():
				continue
			# By construction these gates should be in the on
			# state.  We fake a state transition to off in
			# order to flush the segments
			self._gatehandler(None, timestamp, (segtype, instrument, "off"))
			# But we have to remember to put it back
			self._gatehandler(None, timestamp, (segtype, instrument, "on"))

	def close_segments(self, timestamp):
		"""!
		Record stop times for all open segments and start new ones.

		@param timestamp the time in nanoseconds at which to mark
		the boundary.  If this preceeds and open segment's start
		time, that segment is not closed.
		"""
		with self.dataclass.lock:
			self._close_segments(timestamp)

	def checkpoint(self, timestamp):
		"""!
		Checkpoint, e.g., flush segments and triggers to disk.

		@param timestamp the gstreamer timestamp in nanoseconds of the current buffer in order to close off open segment intervals before writing to disk
		"""
		# FIXME:  the timestamp is used to close off open segments
		# and so should *not* be the timestamp of the current
		# buffer, necessarily, but rather a GPS time guaranteed to
		# precede any future state transitions of any segment list.
		# especially if the pipeline is ever run in an "advanced
		# warning" configuration using the GPS time of a trigger
		# buffer would be an especially bad choice.
		self.flush_segments_to_disk(timestamp)
		try:
			self.dataclass.snapshot_output_file("%s_LLOID" % self.tag, "xml.gz", zero_lag_ranking_stats_filename = self.zero_lag_ranking_stats_filename, verbose = self.verbose)
		except TypeError as te:
			print >>sys.stderr, "Warning: couldn't build output file on checkpoint, probably there aren't any triggers: %s" % te

	def flush_segments_to_disk(self, timestamp):
		"""!
		Flush segments to disk, e.g., when checkpointing or shutting
		down an online pipeline.

		@param timestamp the gstreamer timestamp in nanoseconds of the current buffer in order to close off open segment intervals before writing to disk
		"""
		with self.dataclass.lock:
			try:
				# close out existing and update cumulative segments.
				self._close_segments(timestamp)
				self.update_cumulative_segments()
				ext = segments.segmentlist(seglistdict.extent_all() for seglistdict in self.seglistdicts.values()).extent()
				instruments = set(instrument for seglistdict in self.seglistdicts.values() for instrument in seglistdict)
				#FIXME integrate with the Data class snapshotting directories
				path = str(int(math.floor(ext[0])))[:5]
				try:
					os.mkdir(path)
				except OSError:
					pass
				fname = "%s/%s-%s_SEGMENTS-%d-%d.xml.gz" % (path, "".join(sorted(instruments)), self.tag, int(math.floor(ext[0])), int(math.ceil(ext[1])) - int(math.floor(ext[0])))
				ligolw_utils.write_filename(self.gen_segments_xmldoc(), fname, gz = fname.endswith('.gz'), verbose = self.verbose, trap_signals = None)

				# clear the segment lists in place
				for segtype, seglistdict in self.seglistdicts.items():
					# FIXME:  we don't wipe the
					# triggersegments for now.  the
					# online pipeline needs these to
					# accumulate forever, but that
					# might not be what it should be
					# doing, nor should these
					# necessarily be the segments it
					# uses for livetime.  figure this out
					if segtype == "triggersegments":
						continue
					for seglist in seglistdict.values():
						del seglist[:]
			except ValueError:
				print >>sys.stderr, "Warning: couldn't build segment list on checkpoint, probably there aren't any segments"

	def _gatehandler(self, elem, timestamp, (segtype, instrument, new_state)):
		# FIXME:  this method could conceivably be patched to know
		# what to do with state transitions that appear to have
		# gone backwards, i.e., it could correct segments that are
		# already in the segmentlist.  this might be one way to
		# sort out the problem of segment state snap-shotting code
		# artificially claiming segments to be on beyond the time
		# when they should stop.
		timestamp = LIGOTimeGPS(0, timestamp)	# timestamp is in nanoseconds
		state_key = (segtype, instrument)

		if self.verbose and elem is not None:
			print >>sys.stderr, "%s: %s '%s' state transition: %s @ %s" % (elem.get_name(), instrument, segtype, new_state, str(timestamp))

		# if there is a current_segment_start for this then the
		# state transition has to be off
		if state_key in self.current_segment_start:
			self.seglistdicts[segtype][instrument] |= segments.segmentlist((segments.segment(self.current_segment_start.pop(state_key), timestamp),))
		if new_state == "on":
			self.current_segment_start[state_key] = timestamp
		else:
			assert new_state == "off"

	def gatehandler(self, elem, timestamp, (segtype, instrument, new_state)):
		"""!
		A handler that intercepts gate state transitions. This can set
		the "on" segments for each detector

		@param elem A reference to the lal_gate element or None (only used for verbosity)
		@param timestamp A gstreamer time stamp that marks the state transition (in nanoseconds)
		@param segtype the class of segments this gate is defining, e.g., "datasegments", etc..
		@param instrument the instrument this state transtion is to be attributed to, e.g., "H1", etc..
		@param new_state the state transition, must be either "on" or "off"
		"""
		with self.dataclass.lock:
			self._gatehandler(elem, timestamp, (segtype, instrument, new_state))

	def gen_segments_xmldoc(self):
		"""!
		A method to output the segment list in a valid ligolw xml
		format.
		"""
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})
		with ligolw_segments.LigolwSegments(xmldoc, process) as ligolwsegments:
			for segtype, seglistdict in self.seglistdicts.items():
				ligolwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID snapshot")
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_segments_xml(self):
		"""!
		provide a bottle route to get segment information via a url
		"""
		with self.dataclass.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.gen_segments_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr

	def update_cumulative_segments(self):
		"""!
		A method to update the cumulative segment list 
		"""
		current_gps_time = lal.GPSTimeNow()
		seglist_to_drop = segments.segmentlist([segments.segment(segments.NegInfinity, current_gps_time - self.segment_history_duration)])
		for segtype, seglistdict in self.cumulative_seglistdicts.items():
			seglistdict.extend(self.seglistdicts[segtype])
			seglistdict.coalesce()
			for seglist in seglistdict.values():
				seglist -= seglist_to_drop

	def gen_cumulative_segments_xmldoc(self):
		"""!
		A method to output the cumulative segment list in a valid
		ligolw xml format.
		"""
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})

		# Toggle segments off and on to make sure segment information
		# added to the cumulative segments is current This needs to be
		# run with self.dataclass.lock, but this function is only
		# called currently by web_get_cumulative_segments_xml, which
		# calls with with self.dataclass.lock
		try:
			# FIXME Timestamp here needs to be thought about more,
			# for the same reason mentioned _gatehandler
			timestamp = self.seglistdicts["triggersegments"].extent_all()[1].ns()
			self._close_segments(timestamp)
		except ValueError:
			# no segments
			print >>sys.stderr, "cannot close segments before updating cumulative segments, segment info may be incomplete"
		self.update_cumulative_segments()
		with ligolw_segments.LigolwSegments(xmldoc, process) as ligolwsegments:
			for segtype, seglistdict in self.cumulative_seglistdicts.items():
				ligolwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID snapshot")
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_cumulative_segments_xml(self):
		"""!
		provide a bottle route to get cumulative segment information via a url
		"""
		with self.dataclass.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.gen_cumulative_segments_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr

	def gen_psd_xmldoc(self):
		xmldoc = lal.series.make_psd_xmldoc(self.psds)
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_psd_xml(self):
		with self.dataclass.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.gen_psd_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr


##
# _Gstreamer graph describing this function:_
#
# @dot
# digraph G {
#	rankdir="LR";
#
#	// nodes
#	node [shape=box, style=rounded];
#
# 	lal_firbank [URL="\ref pipeparts.mkfirbank()"];
#	lal_checktimestamps1 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps2 [URL="\ref pipeparts.mkchecktimestamps()"];
#	queue [URL="\ref pipeparts.mkqueue()"];
#	matrixmixer [URL="\ref pipeparts.mkmatrixmixer()", label="lal_matrixmixer\niff bank.mix_matrix", style=filled, color=grey];
#
#	"? sink" -> lal_firbank;
#	lal_firbank -> lal_checktimestamps1;
#
#	// without control
#
#	lal_checktimestamps1 -> queue [label="iff control_snk, control_src are None"];
#	queue -> matrixmixer;
#	matrixmixer -> lal_checktimestamps2;
#	lal_checktimestamps2 -> "? src";
#
#	// with control
#
#	tee [URL="\ref pipeparts.mktee()"];
#	queue2 [URL="\ref pipeparts.mkqueue()"];
#	queue3 [URL="\ref pipeparts.mkqueue()"];
#	queue4 [URL="\ref pipeparts.mkqueue()"];
#	lal_checktimestamps3 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps4 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps5 [URL="\ref pipeparts.mkchecktimestamps()"];
#	capsfilter [URL="\ref pipeparts.mkcapsfilter()"];
#	gate [URL="\ref pipeparts.mkgate()"];
#	"mkcontrolsnksrc()" [URL="\ref mkcontrolsnksrc()"];
#	lal_sumsquares [URL="\ref pipeparts.mksumsquares()"];
#	audioresample [URL="\ref pipeparts.mkresample()"];
#	
#	
#	lal_checktimestamps1 -> tee [label="iff control_snk, control_src are not None"];
#	tee -> lal_sumsquares -> queue2;
#	queue2 -> lal_checktimestamps3;
#	lal_checktimestamps3 -> audioresample;
#	audioresample -> capsfilter;
#	capsfilter -> lal_checktimestamps4;
#	lal_checktimestamps4 -> "mkcontrolsnksrc()"
#	"mkcontrolsnksrc()" -> queue3;
#	queue3 -> gate;
#	tee -> queue4 -> gate;
#	gate -> lal_checktimestamps5;
#	lal_checktimestamps5 -> matrixmixer;
#
# }
# @enddot
#
#
def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, block_duration, nxydump_segment = None, fir_stride = None, control_peak_time = None):
	"""!
	Make a single slice of one branch of the lloid graph, e.g. one instrument and one
	template bank fragment. For details see: http://arxiv.org/abs/1107.2665

	Specifically this implements the filtering of multirate svd basis and
	(conditional) resampling and reconstruction of the physical SNR
	
	@param pipeline The gstreamer pipeline in which to place this graph
	@param src The source of data for this graph provided by a gstreamer element
	@param bank The template bank class
	@param bank_fragment The specific fragment (time slice) of the template bank in question
	@param (control_snk, control_src) An optional tuple of the sink and source elements for a graph that will construct a control time series for the gate which aggregates the orthogonal snrs from each template slice. This is used to conditionally reconstruct the physical SNR of interesting times
	@param gate_attack_length The attack length in samples for the lal_gate element that controls the reconstruction of physical SNRs
	@param gate_hold_length The hold length in samples for the lal_gate element that controls the reconstruction of physical SNRs
	@param block_duration The characteristic buffer size that is passed around, which is useful for constructing queues.
	@param nxydump_segment Not used
	@param fir_stride The target length of output buffers from lal_firbank.  Directly effects latency.  Making this short will force time-domain convolution. Otherwise FFT convolution will be done to save CPU cycles, but at higher latency.
	@param control_peak_time The window over which to find peaks in the control signal.  Shorter windows increase computational cost but probably also detection efficiency.
	"""
	logname = "%s_%.2f.%.2f" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank.  low frequency branches use time-domain
	# convolution, high-frequency branches use FFT convolution with a
	# block stride given by fir_stride.
	#

	latency = -int(round(bank_fragment.start * bank_fragment.rate))
	block_stride = fir_stride * bank_fragment.rate

	# we figure an fft costs ~5 logN flops where N is duration + block
	# stride.  Time domain costs N * block_stride. So if block stride is
	# less than about 5logN you might as well do time domain filtering
	# FIXME This calculation should probably be made more rigorous
	time_domain = 5 * numpy.log2((bank_fragment.end - bank_fragment.start) * bank_fragment.rate + block_stride) > block_stride

	src = pipeparts.mkfirbank(pipeline, src, latency = latency, fir_matrix = bank_fragment.orthogonal_template_bank, block_stride = block_stride, time_domain = time_domain)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank" % logname)
	# uncomment reblock if you ever use really big ffts and want to cut them down a bit
	#src = pipeparts.mkreblock(pipeline, src, block_duration = control_peak_time * gst.SECOND)
	#src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank_reblock" % logname)
	#src = pipeparts.mktee(pipeline, src)	# comment-out the tee below if this is uncommented
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, src), "orthosnr_%s.dump" % logname, segment = nxydump_segment)

	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	if control_snk is not None and control_src is not None:
		src = pipeparts.mktee(pipeline, src)	# comment-out if the tee above is uncommented
		elem = pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights), max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare" % logname)
		elem = pipeline, pipeparts.mkresample(pipeline, elem, quality = 9)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
		elem.link(control_snk)

		#
		# use sum-of-squares aggregate as gate control for orthogonal SNRs
		#
		# FIXME This queue has to be large for the peak finder on the control
		# signal if that element gets smarter maybe this could be made smaller
		# It should be > 1 * control_peak_time * gst.SECOND + 4 * block_duration
		#
		# FIXME for an unknown reason there needs to be an extra large
		# queue in this part of the pipeline in order to prevent
		# lock-ups.  Fortunately this is buffering up relatively
		# lightweight data, i.e., before reconstruction
		#
		# FIXME since peakfinding is done, or control is based on
		# injections only, we ignore the bank.gate_threshold parameter
		# and just use 1e-100

		src = pipeparts.mkgate(
			pipeline,
			pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * (2 * control_peak_time + (abs(gate_attack_length) + abs(gate_hold_length)) / bank_fragment.rate) * Gst.SECOND + 12 * block_duration),
			threshold = 1e-100,
			attack_length = gate_attack_length,
			hold_length = gate_hold_length,
			control = pipeparts.mkqueue(pipeline, control_src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * (2 * control_peak_time + (abs(gate_attack_length) + abs(gate_hold_length)) / bank_fragment.rate) * Gst.SECOND + 12 * block_duration)
		)
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_gate" % logname)

	#
	# reconstruct physical SNRs
	#

	if bank_fragment.mix_matrix is not None:
		src = pipeparts.mkmatrixmixer(pipeline, src, matrix = bank_fragment.mix_matrix)
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_matrixmixer" % logname)

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix

	return src


def mkLLOIDhoftToSnrSlices(pipeline, hoftdict, bank, control_snksrc, block_duration, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, control_peak_time = None, snrslices = None):
	"""!
	Build the pipeline fragment that creates the SnrSlices associated with
	different sample rates from hoft.
	"""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())
	output_rate = max(rates)

	# work out the upsample factors for the attack and hold calculations below
	upsample_factor = dict((rate, rates[i+1] / rate) for i, rate in list(enumerate(rates))[:-1])
	upsample_factor[output_rate] = 0

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	assert autocorrelation_length % 2 == 1
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# loop over template bank slices
	#

	branch_heads = dict((rate, set()) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		# The attack and hold width has three parts
		#
		# 1) The audio resampler filter: 16 comes from the size of
		# the audioresampler filter in samples for the next highest
		# rate at quality 1. Note it must then be converted to the size
		# at the current rate using the upsample_factor dictionary
		# (which is 0 if you are at the max rate).
		#
		# 2) The chisq latency.  You must have at least latency number
		# of points before and after (at the maximum sample rate) to
		# compute the chisq
		#
		# 3) A fudge factor to get the width of the peak.  FIXME this
		# is just set to 1/8th of a second
		peak_half_width = upsample_factor[bank_fragment.rate] * 16 + int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))) + int(math.ceil(bank_fragment.rate / 8.))
		branch_heads[bank_fragment.rate].add(mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(pipeline, hoftdict[bank_fragment.rate], max_size_bytes = 0, max_size_buffers = 0, max_size_time = (1 * fir_stride + int(math.ceil(bank.filter_length))) * Gst.SECOND),
			bank,
			bank_fragment,
			control_snksrc,
			peak_half_width,
			peak_half_width,
			block_duration,
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			control_peak_time = control_peak_time
		))

	#
	# if the calling code has requested copies of the snr
	# slices, sum together the highest sample rate streams and tee
	# them off here.  this needs to be done before constructing the
	# adder network below in order to have access to this snr slice by
	# itself.  if we put this off until later it'll have all the other
	# snrs added into it before we get a chance to tee it off
	#

	if snrslices is not None:
		rate, heads = output_rate, branch_heads[output_rate]
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration) for head in heads))
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads
		branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
		snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# the code below expects an interable of elements
		#

		branch_heads[rate] = set([branch_heads[rate]])

	#
	# sum the snr streams
	#

	if True:	# FIXME:  make conditional on time-slice \chi^{2}
		next_rate = dict(zip(rates, rates[1:]))
	else:
		next_rate = dict((rate, output_rate) for rate in rates if rate != output_rate)

	for rate, heads in sorted(branch_heads.items()):
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = pipeparts.mkadder(pipeline, heads)
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_snr_adder" % (logname, rate))
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads

		#
		# resample if needed
		#

		if rate in next_rate:
			# NOTE: quality = 1 requires that the template
			# slices are padded such that the Nyquist frequency
			# is 1.5 times the highest frequency of the time
			# slice.  NOTE: the adder (that comes downstream of
			# this) isn't quite smart enough to negotiate a
			# common format among its upstream peers so the
			# capsfilter is still required.
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, branch_heads[rate], quality = 1), "audio/x-raw, rate=%d" % next_rate[rate])
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_to_%d_snr_resampler" % (logname, rate, next_rate[rate]))

		#
		# if the calling code has requested copies of the snr
		# slices, tee that off here.  remember that we've already
		# got the highest sample rate slice from above
		#

		if snrslices is not None and rate != output_rate:
			branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
			snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# chain to next adder if this isn't the final answer
		#

		if rate in next_rate:
			branch_heads[next_rate[rate]].add(branch_heads.pop(rate))

	#
	# done
	#

	snr, = branch_heads.values()	# make sure we've summed down to one stream
	return pipeparts.mktogglecomplex(pipeline, snr)


def mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, branch_heads, bank, block_duration):
	"""!
	Build pipeline fragment that computes the TimeSliceChisq from SnrSlices.
	"""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())

	#
	# compute the chifacs for each rate, store in ascending order in rate
	#

	chifacsdict = dict((rate, []) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		chifacsdict[bank_fragment.rate].append(bank_fragment.chifacs)
	chifacs = []
	for rate, facs in sorted(chifacsdict.items()):
		chifacs.append(facs[0][0::2])
		chifacs[-1] += facs[0][1::2]
		for fac in facs[1:]:
			chifacs[-1] += fac[0::2]
			chifacs[-1] += fac[1::2]
		chifacs[-1] /= 2.

	#
	# create timeslicechisq element and add chifacs as a property
	#

	chisq = Gst.ElementFactory.make("lal_timeslicechisq", None)
	pipeline.add(chisq)

	#
	# link the snrslices to the timeslicechisq element in ascending order in rate
	#

	for rate, snrslice in sorted(branch_heads.items()):
		pipeparts.mkqueue(pipeline, snrslice,  max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(chisq)

	#
	# set chifacs-matrix property, needs to be done after snrslices are linked in
	#

	chisq.set_property("chifacs-matrix", chifacs)

	return pipeparts.mkqueue(pipeline, chisq, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)


def mkLLOIDSnrChisqToTriggers(pipeline, snr, chisq, bank, verbose = False, nxydump_segment = None, logname = ""):
	"""!
	Build pipeline fragment that converts single detector SNR and Chisq
	into triggers.
	"""
	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, snr, chisq, template_bank_filename = bank.template_bank_filename, snr_threshold = bank.snr_threshold, sigmasq = bank.sigmasq)
	# FIXME:  add ability to choose this
	# "lal_blcbctriggergen", {"bank-filename": bank.template_bank_filename, "snr-thresh": bank.snr_threshold, "sigmasq": bank.sigmasq}
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % logname)

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = float("inf"), veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 16, control_peak_time = 2, block_duration = Gst.SECOND, reconstruction_segment_list = None):
	"""!
	The multiple instrument, multiple bank LLOID algorithm
	"""

	#
	# check for unrecognized chisq_types, non-unique bank IDs
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError("chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % chisq_type)
	if any(tuple(iterutils.nonuniq(bank.bank_id for bank in banklist)) for banklist in banks.values()):
		raise ValueError("bank IDs are not unique: %s" % "; ".join("for %s: %s" % (instrument, iterutils.nonuniq(bank.bank_id for bank in banklist)) for instrument, banklist in banks.items()))

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	for instrument in detectors.channel_dict:
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates())
		src = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose)
		assert psd_fft_length % 4 == 0, "psd_fft_length (= %g) must be multiple of 4" % psd_fft_length
		hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument] if veto_segments is not None else None, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = psd_fft_length / 4, width = 32)

	#
	# build gate control branches
	#

	if control_peak_time > 0 or reconstruction_segment_list is not None:
		control_branch = {}
		for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
			suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			if instrument != "H2":
				control_branch[(instrument, bank.bank_id)] = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix, reconstruction_segment_list = reconstruction_segment_list, control_peak_samples = control_peak_time * max(bank.get_rates()))
				#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_branch[(instrument, bank.bank_id)][1]), "control_%s.dump" % suffix, segment = nxydump_segment)
	else:
		control_branch = None

	#
	# construct trigger generators
	#

	triggersrcs = dict((instrument, set()) for instrument in hoftdicts)
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if control_branch is not None:
			if instrument != "H2":
				control_snksrc = control_branch[(instrument, bank.bank_id)]
			else:
				control_snksrc = (None, control_branch[("H1", bank.bank_id)][1])
		else:
			control_snksrc = (None, None)
		if chisq_type == 'timeslicechisq':
			snrslices = {}
		else:
			snrslices = None
		snr = mkLLOIDhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			control_snksrc,
			block_duration,
			verbose = verbose,
			logname = suffix,
			nxydump_segment = nxydump_segment,
			control_peak_time = control_peak_time,
			fir_stride = fir_stride,
			snrslices = snrslices
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)
		# uncomment this tee if the diagnostic sinks below are
		# needed
		#snr = pipeparts.mktee(pipeline, snr)
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is 1 second at max rate, ie max(rates)
			head = pipeparts.mkitac(pipeline, snr, 1 * max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = bank.autocorrelation_mask, snr_thresh = bank.snr_threshold, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs[instrument].add(head)
		else:
			raise NotImplementedError("Currently only 'autochisq' is supported")
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert any(triggersrcs.values())
	return triggersrcs
