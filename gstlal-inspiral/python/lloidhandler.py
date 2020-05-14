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
# gstlal_inspiral's GStreamer pipeline handler.
#
#
# Review Status
#
# migrated from lloidparts.py 2018-06-15
#
# | Names                                 | Hash                                     | Date       | Diff to Head of Master      |
# | ------------------------------------- | ---------------------------------------- | ---------- | --------------------------- |
# | Sathya, Duncan Me, Jolien, Kipp, Chad | 2f5f73f15a1903dc7cc4383ef30a4187091797d1 | 2014-05-02 | <a href="@gstlal_inspiral_cgit_diff/python/lloidparts.py?id=HEAD&id2=2f5f73f15a1903dc7cc4383ef30a4187091797d1">lloidparts.py</a> |
#
#

##
# @package lloidparts
#
# gstlal_inspiral's GStreamer pipeline handler.
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from collections import defaultdict, deque
try:
	from fpconst import NaN
	from fpconst import PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	PosInf = float("+inf")
import itertools
import math
import numpy
import os
import resource
from scipy.interpolate import interp1d
import io
import sys
import threading
import time
import shutil
import json


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


from ligo.lw import ligolw
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process
from ligo.lw.utils import segments as ligolw_segments
from gstlal import bottle
from gstlal import far
from gstlal import inspiral
from gstlal import p_astro_gstlal
from gstlal import pipeio
from gstlal import simplehandler
from gstlal import streamthinca
from gstlal.snglinspiraltable import GSTLALSnglInspiral as SnglInspiral
import lal
from lal import LIGOTimeGPS
from lal import rate
from lal.utils import CacheEntry
from ligo import segments
from ligo.segments import utils as segmentsUtils


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def message_new_checkpoint(src, timestamp = None):
	s = Gst.Structure.new_empty("CHECKPOINT")
	message = Gst.Message.new_application(src, s)
	if timestamp is not None:
		message.timestamp = timestamp
	return message


def subdir_from_T050017_filename(fname):
	path = str(CacheEntry.from_T050017(fname).segment[0])[:5]
	try:
		os.mkdir(path)
	except OSError:
		pass
	return path


#
# =============================================================================
#
#                                Web Eye Candy
#
# =============================================================================
#


class EyeCandy(object):
	def __init__(self, instruments, kafka_server, tag, pipeline, segmentstracker):
		self.kafka_server = kafka_server
		self.tag = tag
		self.gate_history = segmentstracker.gate_history
		self.latency_histogram = rate.BinnedArray(rate.NDBins((rate.LinearPlusOverflowBins(5, 205, 22),)))
		# NOTE most of this data is collected at 1Hz, thus a 300
		# element deque should hold about 5 minutes of history.
		# Keeping the deque short is desirable for efficiency in
		# downloads, but it could mean that data is lost (though the
		# snapshot every ~4 hours will contain the information in
		# general)
		self.latency_history = deque(maxlen = 300)
		self.snr_history = deque(maxlen = 300)
		self.likelihood_history = deque(maxlen = 300)
		self.far_history = deque(maxlen = 300)
		self.ram_history = deque(maxlen = 2)
		self.ifo_snr_history = dict((instrument, deque(maxlen = 300)) for instrument in instruments)
		self.strain = {}
		for instrument in instruments:
			name = "%s_strain_audiorate" % instrument
			elem = pipeline.get_by_name(name)
			if elem is not None:
				self.strain[instrument] = elem
		self.time_since_last_state = None

		#
		# setup bottle routes
		#

		bottle.route("/latency_histogram.txt")(self.web_get_latency_histogram)
		bottle.route("/latency_history.txt")(self.web_get_latency_history)
		bottle.route("/snr_history.txt")(self.web_get_snr_history)
		if "H1" in instruments:
			bottle.route("/H1_snr_history.txt")(self.web_get_H1_snr_history)
		if "L1" in instruments:
			bottle.route("/L1_snr_history.txt")(self.web_get_L1_snr_history)
		if "V1" in instruments:
			bottle.route("/V1_snr_history.txt")(self.web_get_V1_snr_history)
		bottle.route("/likelihood_history.txt")(self.web_get_likelihood_history)
		bottle.route("/far_history.txt")(self.web_get_far_history)
		bottle.route("/ram_history.txt")(self.web_get_ram_history)

		#
		# Setup kafka producer
		#

		if self.kafka_server is not None:
			from kafka import KafkaProducer
			self.producer = KafkaProducer(
				bootstrap_servers=[self.kafka_server],
				key_serializer=lambda m: json.dumps(m).encode('utf-8'),
				value_serializer=lambda m: json.dumps(m).encode('utf-8'),
			)
		else:
			self.producer = None

		# FIXME, it is silly to store kafka data like this since we
		# have all the other data structures, but since we are also
		# maintaining the bottle route methods, we should keep this a
		# bit separate for now to not disrupt too much.

		self.kafka_data = defaultdict(lambda: {'time': [], 'data': []})
		self.kafka_data["coinc"] = []

	def update(self, events, last_coincs):
		self.ram_history.append((float(lal.UTCToGPS(time.gmtime())), (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1048576.)) # GB
		if events:
			maxevents = {}
			for event in events:
				if (event.ifo not in maxevents) or (event.snr > maxevents[event.ifo].snr):
					maxevents[event.ifo] = event
			for ifo, event in maxevents.items():
				t, snr = float(event.end), event.snr
				self.ifo_snr_history[ifo].append((t, snr))
				if self.producer is not None:
					self.kafka_data["%s_snr_history" % ifo]["time"].append(t)
					self.kafka_data["%s_snr_history" % ifo]["data"].append(snr)
		if last_coincs:
			coinc_inspiral_index = last_coincs.coinc_inspiral_index
			coinc_event_index = last_coincs.coinc_event_index
			sngl_inspiral_index = last_coincs.sngl_inspiral_index
			coinc_dict_list = []
			for coinc_event_id in coinc_event_index:
				coinc_dict = {}
				for attr in ("combined_far", "snr", "false_alarm_rate"):
					try:
						coinc_dict[attr] = float(getattr(coinc_inspiral_index[coinc_event_id], attr))
					except TypeError as e:
						pass#print >>sys.stderr, e, attr, getattr(coinc_inspiral_index[coinc_event_id], attr)
				coinc_dict["end"] = float(coinc_inspiral_index[coinc_event_id].end)
				for attr in ("likelihood",):
					try:
						coinc_dict[attr] = float(getattr(coinc_event_index[coinc_event_id], attr))
					except TypeError as e:
						pass#print >>sys.stderr, e, attr, getattr(coinc_event_index[coinc_event_id], attr)
				for sngl_row in sngl_inspiral_index[coinc_event_id]:
					for attr in ("snr", "chisq", "mass1", "mass2", "spin1z", "spin2z", "coa_phase"):
						coinc_dict["%s_%s" % (sngl_row.ifo, attr)] = float(getattr(sngl_row, attr))
					coinc_dict["%s_end" % sngl_row.ifo] = float(sngl_row.end)
				coinc_dict_list.append(coinc_dict)
			self.kafka_data["coinc"].extend(coinc_dict_list)
			for coinc_inspiral in coinc_inspiral_index.values():
				# latency in .minimum_duration
				# FIXME:  update when a proper column is available
				self.latency_histogram[coinc_inspiral.minimum_duration,] += 1
			# latency in .minimum_duration
			# FIXME:  update when a proper column is available
			max_latency, max_latency_t = max((coinc_inspiral.minimum_duration, float(coinc_inspiral.end)) for coinc_inspiral in coinc_inspiral_index.values())
			self.latency_history.append((max_latency_t, max_latency))

			max_snr, max_snr_t = max((coinc_inspiral.snr, float(coinc_inspiral.end)) for coinc_inspiral in coinc_inspiral_index.values())
			self.snr_history.append((max_snr_t, max_snr))

			max_likelihood, max_likelihood_t, max_likelihood_far = max((coinc_event_index[coinc_event_id].likelihood, float(coinc_inspiral.end), coinc_inspiral.combined_far) for coinc_event_id, coinc_inspiral in coinc_inspiral_index.items())
			if max_likelihood is not None:
				self.likelihood_history.append((max_likelihood_t, max_likelihood))
			if max_likelihood_far is not None:
				self.far_history.append((max_likelihood_t, max_likelihood_far))

			if self.producer is not None:
				for ii, column in enumerate(["time", "data"]):
					self.kafka_data["latency_history"][column].append(float(self.latency_history[-1][ii]))
					self.kafka_data["snr_history"][column].append(float(self.snr_history[-1][ii]))
				if max_likelihood is not None:
					self.kafka_data["likelihood_history"]["time"].append(float(max_likelihood_t))
					self.kafka_data["likelihood_history"]["data"].append(float(max_likelihood))
				if max_likelihood_far is not None:
					self.kafka_data["far_history"]["time"].append(float(max_likelihood_t))
					self.kafka_data["far_history"]["data"].append(float(max_likelihood_far))

		t = inspiral.now()
		if self.time_since_last_state is None:
			self.time_since_last_state = t

		# send state/segment information to kafka every second
		if self.producer is not None and (t - self.time_since_last_state) >= 1:
			self.time_since_last_state = t
			for ii, column in enumerate(["time", "data"]):
				self.kafka_data["ram_history"][column].append(float(self.ram_history[-1][ii]))

			# collect gate segments
			for gate in self.gate_history.keys():
				for instrument, seg_history in self.gate_history[gate].items():
					if not seg_history:
						continue

					# get on/off points, add point at +inf
					gate_interp_times, gate_interp_onoff = zip(*seg_history)
					gate_interp_times = list(gate_interp_times)
					gate_interp_times.append(2000000000)
					gate_interp_onoff = list(gate_interp_onoff)
					gate_interp_onoff.append(gate_interp_onoff[-1])

					# regularly sample from on/off points
					gate_times = numpy.arange(int(self.time_since_last_state), int(t + 1), 0.25)
					gate_onoff = interp1d(gate_interp_times, gate_interp_onoff, kind='zero')(gate_times)
					self.kafka_data["%s_%s" % (instrument, gate)]["time"].extend([t for t in gate_times if t >= self.time_since_last_state])
					self.kafka_data["%s_%s" % (instrument, gate)]["data"].extend([state for t, state in zip(gate_times, gate_onoff) if t >= self.time_since_last_state])

			# collect strain dropped samples
			for instrument, elem in self.strain.items():
				# I know the name is strain_drop even though it
				# comes from the "add" property. that is
				# because audiorate has to "add" samples when
				# data is dropped.
				# FIXME don't hard code the rate
				self.kafka_data["%s_strain_dropped" % instrument]["time"].append(float(t))
				self.kafka_data["%s_strain_dropped" % instrument]["data"].append(elem.get_property("add") / 16384.)

			# Send all of the kafka messages and clear the data
			#self.producer.send(self.tag, self.kafka_data)
			for route in self.kafka_data.keys():
				self.producer.send(route, key=self.tag, value=self.kafka_data[route])
			# This line forces the send but is blocking!! not the
			# best idea for production running since we value
			# latency over getting metric data out
			#self.producer.flush()
			for route in self.kafka_data.keys():
				self.kafka_data[route] = {'time': [], 'data': []}
			self.kafka_data["coinc"] = []

	def web_get_latency_histogram(self):
		with self.lock:
			for latency, number in zip(self.latency_histogram.centres()[0][1:-1], self.latency_histogram.array[1:-1]):
				yield "%e %e\n" % (latency, number)

	def web_get_latency_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, latency in self.latency_history:
				yield "%f %e\n" % (time, latency)

	def web_get_snr_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.snr_history:
				yield "%f %e\n" % (time, snr)

	def web_get_H1_snr_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.ifo_snr_history["H1"]:
				yield "%f %e\n" % (time, snr)

	def web_get_L1_snr_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.ifo_snr_history["L1"]:
				yield "%f %e\n" % (time, snr)

	def web_get_V1_snr_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, snr in self.ifo_snr_history["V1"]:
				yield "%f %e\n" % (time, snr)

	def web_get_likelihood_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, like in self.likelihood_history:
				yield "%f %e\n" % (time, like)

	def web_get_far_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, far in self.far_history:
				yield "%f %e\n" % (time, far)

	def web_get_ram_history(self):
		with self.lock:
			# first one in the list is sacrificed for a time stamp
			for time, ram in self.ram_history:
				yield "%f %e\n" % (time, ram)


#
# =============================================================================
#
#                             Segmentlist Tracker
#
# =============================================================================
#


class SegmentsTracker(object):
	def __init__(self, pipeline, instruments, segment_history_duration = LIGOTimeGPS(2592000), verbose = False):
		self.lock = threading.Lock()
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
			"dqvectorsegments": "dq_vector_gate",
			"whitehtsegments": "ht_gate"
		}

		# dictionary mapping segtype to segmentlist dictionary
		# mapping instrument to segment list
		self.seglistdicts = dict((segtype, segments.segmentlistdict((instrument, segments.segmentlist()) for instrument in instruments)) for segtype in gate_suffix)

		# create a copy to keep track of recent segment history
		self.recent_segment_histories = self.seglistdicts.copy()
		self.segment_history_duration = segment_history_duration

		# recent gate history encoded in on/off bits
		self.gate_history = {segtype: {instrument: deque(maxlen = 20) for instrument in instruments} for segtype in gate_suffix}

		# iterate over segment types and instruments, look for the
		# gate element that should provide those segments, and
		# connect handlers to collect the segments
		if verbose:
			print(sys.stderr, "connecting segment handlers to gates ...", file=sys.stderr)
		for segtype, seglistdict in self.seglistdicts.items():
			for instrument in seglistdict:
				try:
					name = "%s_%s" % (instrument, gate_suffix[segtype])
				except KeyError:
					# this segtype doesn't come from
					# gate elements
					continue
				elem = pipeline.get_by_name(name)
				if elem is None:
					# ignore missing gate elements
					if verbose:
						print("\tcould not find %s for %s '%s'" % (name, instrument, segtype), file=sys.stderr)
					continue
				if verbose:
					print("\tfound %s for %s '%s'" % (name, instrument, segtype), file=sys.stderr)
				elem.connect("start", self.gatehandler, (segtype, instrument, "on"))
				elem.connect("stop", self.gatehandler, (segtype, instrument, "off"))
				elem.set_property("emit-signals", True)
		if verbose:
			print("... done connecting segment handlers to gates", file=sys.stderr)


	def __gatehandler(self, elem, timestamp, seg_state_input):
		"""!
		A handler that intercepts gate state transitions.

		@param elem A reference to the lal_gate element or None
		(only used for verbosity)
		@param timestamp A gstreamer time stamp (integer
		nanoseconds) that marks the state transition
		@param segtype the class of segments this gate is defining,
		e.g., "datasegments", etc..
		@param instrument the instrument this state transtion is to
		be attributed to, e.g., "H1", etc..
		@param new_state the state transition, must be either "on"
		or "off"

		Must be called with the lock held.
		"""
		# convert integer nanoseconds to LIGOTimeGPS
		timestamp = LIGOTimeGPS(0, timestamp)

		# unpack argument tuple:
		segtype, instrument, new_state = seg_state_input

		if self.verbose:
			print("%s: %s '%s' state transition: %s @ %s" % ((elem.get_name() if elem is not None else "<internal>"), instrument, segtype, new_state, str(timestamp)), file= sys.stderr)

		if new_state == "off":
			# record end of segment
			self.seglistdicts[segtype][instrument] -= segments.segmentlist((segments.segment(timestamp, segments.PosInfinity),))
			self.gate_history[segtype][instrument].append((float(timestamp), 0.))
		elif new_state == "on":
			# record start of new segment
			self.seglistdicts[segtype][instrument] += segments.segmentlist((segments.segment(timestamp, segments.PosInfinity),))
			self.gate_history[segtype][instrument].append((float(timestamp), 1.))
		else:
			assert False, "impossible new_state '%s'" % new_state


	def gatehandler(self, elem, timestamp, seg_state_input):
		segtype, instrument, new_state = seg_state_input
		with self.lock:
			self.__gatehandler(elem, timestamp, (segtype, instrument, new_state))


	def terminate(self, timestamp):
		# terminate all segments
		with self.lock:
			for segtype, seglistdict in self.seglistdicts.items():
				for instrument in seglistdict:
					self.__gatehandler(None, timestamp, (segtype, instrument, "off"))


	def gen_segments_xmldoc(self):
		"""!
		A method to output the segment list in a valid ligolw xml
		format.

		Must be called with the lock held.
		"""
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})
		with ligolw_segments.LigolwSegments(xmldoc, process) as ligolwsegments:
			for segtype, seglistdict in self.seglistdicts.items():
				ligolwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID snapshot")
		ligolw_process.set_process_end_time(process)
		return xmldoc


	def __T050017_filename(self, description, extension):
		"""!
		Must be called with the lock held.
		"""
		# input check
		if "-" in description:
			raise ValueError("invalid characters in description '%s'" % description)

		# determine current extent
		instruments = set(instrument for seglistdict in self.seglistdicts.values() for instrument in seglistdict)
		segs = segments.segmentlist(seglistdict.extent_all() for seglistdict in self.seglistdicts.values() if any(seglistdict.values()))
		if segs:
			start, end = segs.extent()
			if math.isinf(end):
				end = inspiral.now()
		else:
			# silence errors at start-up.
			# FIXME:  this is probably dumb.  who cares.
			start = end = inspiral.now()

		# construct and return filename
		start, end = int(math.floor(start)), int(math.ceil(end))
		return "%s-%s-%d-%d.%s" % ("".join(sorted(instruments)), description, start, end - start, extension)


	def T050017_filename(self, description, extension):
		with self.lock:
			return self.__T050017_filename(description, extension)


	def flush_segments_to_disk(self, tag, timestamp):
		"""!
		Flush segments to disk, e.g., when checkpointing or
		shutting down an online pipeline.

		@param timestamp the LIGOTimeGPS timestamp of the current
		buffer in order to close off open segment intervals before
		writing to disk
		"""
		with self.lock:
			# make a copy of the current segmentlistdicts
			seglistdicts = dict((key, value.copy()) for key, value in self.seglistdicts.items())

			# keep everything before timestamp in the current
			# segmentlistdicts.  keep everything after
			# timestamp in the copy.  we need to apply the cut
			# this way around so that the T050017 filename
			# constructed below has the desired start and
			# duration
			for seglistdict in self.seglistdicts.values():
				seglistdict -= seglistdict.fromkeys(seglistdict, segments.segmentlist([segments.segment(timestamp, segments.PosInfinity)]))
			for seglistdict in seglistdicts.values():
				seglistdict -= seglistdict.fromkeys(seglistdict, segments.segmentlist([segments.segment(segments.NegInfinity, timestamp)]))

			# write the current (clipped) segmentlistdicts to
			# disk
			fname = self.__T050017_filename("%s_SEGMENTS" % tag, "xml.gz")
			fname = os.path.join(subdir_from_T050017_filename(fname), fname)
			ligolw_utils.write_filename(self.gen_segments_xmldoc(), fname, gz = fname.endswith('.gz'), verbose = self.verbose, trap_signals = None)

			# continue with the (clipped) copy
			self.seglistdicts = seglistdicts


	def web_get_segments_xml(self):
		"""!
		provide a bottle route to get segment information via a url
		"""
		with self.lock:
			output = io.StringIO()
			ligolw_utils.write_fileobj(self.gen_segments_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr


	def update_recent_segment_history(self):
		"""!
		A method to update the recent segment histories

		Must be called with the lock held.
		"""
		current_gps_time = lal.GPSTimeNow()
		interval_to_discard = segments.segmentlist((segments.segment(segments.NegInfinity, current_gps_time - self.segment_history_duration),))
		for segtype, seglistdict in self.recent_segment_histories.items():
			seglistdict.extend(self.seglistdicts[segtype])
			seglistdict.coalesce()
			for seglist in seglistdict.values():
				seglist -= interval_to_discard


	def gen_recent_segment_history_xmldoc(self):
		"""!
		Construct and return a LIGOLW XML tree containing the
		recent segment histories.

		Must be called with the lock held.
		"""
		self.update_recent_segment_history()
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {})
		with ligolw_segments.LigolwSegments(xmldoc, process) as ligolwsegments:
			for segtype, seglistdict in self.recent_segment_histories.items():
				ligolwsegments.insert_from_segmentlistdict(seglistdict, name = segtype, comment = "LLOID snapshot")
		ligolw_process.set_process_end_time(process)
		return xmldoc


	def web_get_recent_segment_history_xml(self):
		"""!
		provide a bottle route to get recent segment history
		information via a url
		"""
		with self.lock:
			output = io.StringIO()
			ligolw_utils.write_fileobj(self.gen_recent_segment_history_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr


#
# =============================================================================
#
#                               Pipeline Handler
#
# =============================================================================
#


class Handler(simplehandler.Handler):
	"""!
	A subclass of simplehandler.Handler to be used with e.g.,
	gstlal_inspiral

	Implements additional message handling for dealing with spectrum
	messages and checkpoints for the online analysis including periodic
	dumps of segment information, trigger files and background
	distribution statistics.
	"""
	def __init__(self, mainloop, pipeline, coincs_document, rankingstat, horizon_distance_func, gracedbwrapper, zerolag_rankingstatpdf_url = None, rankingstatpdf_url = None, ranking_stat_output_url = None, ranking_stat_input_url = None, likelihood_snapshot_interval = None, sngls_snr_threshold = None, tag = "", kafka_server = "10.14.0.112:9092", cluster = False, cap_singles = False, FAR_trialsfactor = 1.0, activation_counts = None, verbose = False):
		"""!
		@param mainloop The main application's event loop
		@param pipeline The gstreamer pipeline that is being
		controlled by this handler
		@param dataclass A Data class instance
		@param tag The tag to use for naming file snapshots, e.g.
		the description will be "%s_LLOID" % tag
		@param verbose Be verbose
		"""
		super(Handler, self).__init__(mainloop, pipeline)

		#
		# initialize
		#

		self.lock = threading.Lock()
		self.coincs_document = coincs_document
		self.pipeline = pipeline
		self.tag = tag
		self.verbose = verbose
		# None to disable periodic snapshots, otherwise seconds
		self.likelihood_snapshot_interval = likelihood_snapshot_interval
		self.likelihood_snapshot_timestamp = None
		self.cluster = cluster
		self.cap_singles = cap_singles
		self.FAR_trialsfactor = FAR_trialsfactor
		self.activation_counts = activation_counts

		self.gracedbwrapper = gracedbwrapper
		# FIXME:   detangle this
		self.gracedbwrapper.lock = self.lock

		#
		# setup segment list collection from gates
		#

		self.segmentstracker = SegmentsTracker(pipeline, rankingstat.instruments, verbose = verbose)

		#
		# set up metric collection
		#

		self.eye_candy = EyeCandy(rankingstat.instruments, kafka_server, self.tag, pipeline, self.segmentstracker)
		# FIXME:   detangle this
		self.eye_candy.lock = self.lock

		#
		# setup bottle routes (and rahwts)
		#

		bottle.route("/cumulative_segments.xml")(self.segmentstracker.web_get_recent_segment_history_xml)
		bottle.route("/psds.xml")(self.web_get_psd_xml)
		bottle.route("/rankingstat.xml")(self.web_get_rankingstat)
		bottle.route("/segments.xml")(self.segmentstracker.web_get_segments_xml)
		bottle.route("/sngls_snr_threshold.txt", method = "GET")(self.web_get_sngls_snr_threshold)
		bottle.route("/sngls_snr_threshold.txt", method = "POST")(self.web_set_sngls_snr_threshold)
		bottle.route("/zerolag_rankingstatpdf.xml")(self.web_get_zerolag_rankingstatpdf)

		#
		# attach a StreamThinca instance to ourselves
		#

		self.stream_thinca = streamthinca.StreamThinca(
			coincs_document.xmldoc,
			coincs_document.process_id,
			delta_t = rankingstat.delta_t,
			min_instruments = rankingstat.min_instruments,
			sngls_snr_threshold = sngls_snr_threshold
		)

		#
		# setup likelihood ratio book-keeping.
		#
		# in online mode, if ranking_stat_input_url is set then on
		# each snapshot interval, and before providing stream
		# thinca with its ranking statistic information, the
		# current rankingstat object is replaced with the contents
		# of that file.  this is intended to be used by trigger
		# generator jobs on the injection branch of an online
		# analysis to import ranking statistic information from
		# their non-injection cousins instead of using whatever
		# statistics they've collected internally.
		# ranking_stat_input_url is not used when running offline.
		#
		# ranking_stat_output_url provides the name of the file to
		# which the internally-collected ranking statistic
		# information is to be written whenever output is written
		# to disk.  if set to None, then only the trigger file will
		# be written, no ranking statistic information will be
		# written.  normally it is set to a non-null value, but
		# injection jobs might be configured to disable ranking
		# statistic output since they produce nonsense.
		#

		self.ranking_stat_input_url = ranking_stat_input_url
		self.ranking_stat_output_url = ranking_stat_output_url
		self.rankingstat = rankingstat

		#
		# if we have been supplied with external ranking statistic
		# information then use it to enable ranking statistic
		# assignment in streamthinca.
		#

		if self.ranking_stat_input_url is not None:
			if self.rankingstat.is_healthy(self.verbose):
				self.stream_thinca.ln_lr_from_triggers = far.OnlineFrankensteinRankingStat(self.rankingstat, self.rankingstat).finish().ln_lr_from_triggers
				if self.verbose:
					print("ranking statistic assignment ENABLED", file=sys.stderr)
			else:
				self.stream_thinca.ln_lr_from_triggers = None
				if self.verbose:
					print("ranking statistic assignment DISABLED", file=sys.stderr)
		elif False:
			# FIXME:  move sum-of-SNR^2 cut into this object's
			# .__call__() and then use as coinc sieve function
			# instead.  left here temporariliy to remember how
			# to initialize it
			self.stream_thinca.ln_lr_from_triggers = far.DatalessRankingStat(
				template_ids = rankingstat.template_ids,
				instruments = rankingstat.instruments,
				min_instruments = rankingstat.min_instruments,
				delta_t = rankingstat.delta_t
			).finish().ln_lr_from_triggers
			if self.verbose:
				print("ranking statistic assignment ENABLED", file=sys.stderr)
		else:
			self.stream_thinca.ln_lr_from_triggers = None
			if self.verbose:
				print("ranking statistic assignment DISABLED", file=sys.stderr)

		#
		# zero_lag_ranking_stats is a RankingStatPDF object that is
		# used to accumulate a histogram of the likelihood ratio
		# values assigned to zero-lag candidates.  this is required
		# to implement the extinction model for low-significance
		# events during online running but otherwise is optional.
		#
		# FIXME:  if the file does not exist or is not readable,
		# the code silently initializes a new, empty, histogram.
		# it would be better to determine whether or not the file
		# is required and fail when it is missing
		#

		if zerolag_rankingstatpdf_url is not None and os.access(ligolw_utils.local_path_from_url(zerolag_rankingstatpdf_url), os.R_OK):
			_, self.zerolag_rankingstatpdf = far.parse_likelihood_control_doc(ligolw_utils.load_url(zerolag_rankingstatpdf_url, verbose = verbose, contenthandler = far.RankingStat.LIGOLWContentHandler))
			if self.zerolag_rankingstatpdf is None:
				raise ValueError("\"%s\" does not contain ranking statistic PDF data" % zerolag_rankingstatpdf_url)
		elif zerolag_rankingstatpdf_url is not None:
			# initialize an all-zeros set of PDFs
			self.zerolag_rankingstatpdf = far.RankingStatPDF(rankingstat, nsamples = 0)
		else:
			self.zerolag_rankingstatpdf = None
		self.zerolag_rankingstatpdf_url = zerolag_rankingstatpdf_url

		#
		# set horizon distance calculator
		#

		self.horizon_distance_func = horizon_distance_func

		#
		# rankingstatpdf contains the RankingStatPDF object (loaded
		# from rankingstatpdf_url) used to initialize the FAPFAR
		# object for on-the-fly FAP and FAR assignment.  except to
		# initialize the FAPFAR object it is not used for anything,
		# but is retained so that it can be exposed through the web
		# interface for diagnostic purposes and uploaded to gracedb
		# with candidates.  the extinction model is applied to
		# initialize the FAPFAR object but the original is retained
		# for upload to gracedb, etc.
		#

		self.rankingstatpdf_url = rankingstatpdf_url
		self.load_rankingstat_pdf()

		#
		# most recent PSDs
		#

		self.psds = {}

		#
		# use state vector elements to 0 horizon distances when
		# instruments are off
		#

		if verbose:
			print("connecting horizon distance handlers to gates ...", file=sys.stderr)
		self.absent_instruments = set()
		for instrument in rankingstat.instruments:
			name = "%s_ht_gate" % instrument
			elem = pipeline.get_by_name(name)
			if elem is None:
				# FIXME:  if there is no data for an
				# instrument for which we have ranking
				# statistic data then the horizon distance
				# record needs to indicate that it was off
				# for the entire segment.  should probably
				# 0 the horizon distance history at start
				# and stop of each stream, but there is a
				# statement in the EOS handler that we
				# don't do that, and I can remember why
				# that is.
				self.absent_instruments.add(instrument)
				continue
				raise ValueError("cannot find \"%s\" element for %s" % (name, instrument))
			if verbose:
				print("\tfound %s for %s" % (name, instrument), file=sys.stderr)
			elem.connect("start", self.horizgatehandler, (instrument, True))
			elem.connect("stop", self.horizgatehandler, (instrument, False))
			elem.set_property("emit-signals", True)
		if verbose:
			print("... done connecting horizon distance handlers to gates", file=sys.stderr)


	def do_on_message(self, bus, message):
		"""!
		Handle application-specific message types, e.g., spectrum
		and checkpointing messages.

		@param bus A reference to the pipeline's bus
		@param message A reference to the incoming message
		"""
		#
		# return value of True tells parent class that we have done
		# all that is needed in response to the message, and that
		# it should ignore it.  a return value of False means the
		# parent class should do what it thinks should be done
		#
		if message.type == Gst.MessageType.ELEMENT:
			if message.get_structure().get_name() == "spectrum":
				# get the instrument, psd, and timestamp.
				# the "stability" is a measure of the
				# fraction of the configured averaging
				# timescale used to obtain this
				# measurement.
				# NOTE: epoch is used for the timestamp,
				# this is the middle of the most recent FFT
				# interval used to obtain this PSD
				instrument = message.src.get_name().split("_")[-1]
				psd = pipeio.parse_spectrum_message(message)
				timestamp = psd.epoch
				stability = float(message.src.get_property("n-samples")) / message.src.get_property("average-samples")

				# save
				self.psds[instrument] = psd

				# update horizon distance history.  if the
				# whitener's average is not satisfactorily
				# converged, claim the horizon distance is
				# 0 (equivalent to claiming the instrument
				# to be off at this time).  this has the
				# effect of vetoing candidates from these
				# times.
				if stability > 0.3:
					horizon_distance = self.horizon_distance_func(psd, 8.0)[0]
					assert not (math.isnan(horizon_distance) or math.isinf(horizon_distance))
				else:
					horizon_distance = 0.
				self.record_horizon_distance(instrument, float(timestamp), horizon_distance)
				return True
		elif message.type == Gst.MessageType.APPLICATION:
			if message.get_structure().get_name() == "CHECKPOINT":
				self.checkpoint(LIGOTimeGPS(0, message.timestamp))
				return True
		elif message.type == Gst.MessageType.EOS:
			with self.lock:
				# FIXME:  how to choose correct timestamp?
				# note that EOS messages' timestamps are
				# set to CLOCK_TIME_NONE so they can't be
				# used for this.
				try:
					# seconds
					timestamp = self.rankingstat.segmentlists.extent_all()[1]
				except ValueError:
					# no segments
					return False
				# convert to integer nanoseconds
				timestamp = LIGOTimeGPS(timestamp).ns()
				# terminate all segments
				self.segmentstracker.terminate(timestamp)
				# NOTE:  we do not zero the horizon
				# distances in the ranking statistic at EOS
			# NOTE:  never return True from the EOS code-path,
			# so as to not stop the parent class from doing
			# EOS-related stuff
			return False
		return False


	def load_rankingstat_pdf(self):
		# FIXME:  if the file can't be accessed the code silently
		# disables FAP/FAR assignment.  need to figure out when
		# failure is OK and when it's not OK and put a better check
		# here.
		if self.rankingstatpdf_url is not None and os.access(ligolw_utils.local_path_from_url(self.rankingstatpdf_url), os.R_OK):
			_, self.rankingstatpdf = far.parse_likelihood_control_doc(ligolw_utils.load_url(self.rankingstatpdf_url, verbose = self.verbose, contenthandler = far.RankingStat.LIGOLWContentHandler))
			if self.rankingstatpdf is None:
				raise ValueError("\"%s\" does not contain ranking statistic PDFs" % url)
			if not self.rankingstat.template_ids <= self.rankingstatpdf.template_ids:
				raise ValueError("\"%s\" is for the wrong templates")
			if self.rankingstatpdf.is_healthy(self.verbose):
				self.fapfar = far.FAPFAR(self.rankingstatpdf.new_with_extinction())
				if self.verbose:
					print("false-alarm probability and rate assignment ENABLED", file=sys.stderr)
			else:
				self.fapfar = None
				if self.verbose:
					print("false-alarm probability and rate assignment DISABLED", file=sys.stderr)
		else:
			self.rankingstatpdf = None
			self.fapfar = None


	def appsink_new_buffer(self, elem):
		with self.lock:
			# retrieve triggers from appsink element
			buf = elem.emit("pull-sample").get_buffer()
			events = []
			for i in range(buf.n_memory()):
				memory = buf.peek_memory(i)
				result, mapinfo = memory.map(Gst.MapFlags.READ)
				assert result
				# NOTE NOTE NOTE NOTE
				# It is critical that the correct class'
				# .from_buffer() method be used here.  This
				# code is interpreting the buffer's
				# contents as an array of C structures and
				# building instances of python wrappers of
				# those structures but if the python
				# wrappers are for the wrong structure
				# declaration then terrible terrible things
				# will happen
				# NOTE NOTE NOTE NOTE
				# FIXME why does mapinfo.data come out as
				# an empty list on some occasions???
				if mapinfo.data:
					events.extend(SnglInspiral.from_buffer(mapinfo.data))
				memory.unmap(mapinfo)

			# FIXME:  ugly way to get the instrument
			instruments = set([event.ifo for event in events])

			# FIXME calculate a chisq weighted SNR and store it in the bank_chisq column
			for event in events:
				event.bank_chisq = event.snr / ((1 + max(1., event.chisq)**3)/2.0)**(1./5.)

			# extract segment.  move the segment's upper
			# boundary to include all triggers.  ARGH the 1 ns
			# offset is needed for the respective trigger to be
			# "in" the segment (segments are open from above)
			# FIXME:  is there another way?
			buf_timestamp = LIGOTimeGPS(0, buf.pts)
			buf_seg = dict((instrument, segments.segment(buf_timestamp, max(buf_timestamp + LIGOTimeGPS(0, buf.duration), max(event.end for event in events if event.ifo == instrument) + 0.000000001))) for instrument in instruments)
			buf_is_gap = bool(buf.mini_object.flags & Gst.BufferFlags.GAP)
			# sanity check that gap buffers are empty
			assert not (buf_is_gap and events)

			# safety check end times.  we cannot allow triggr
			# times to go backwards.  they cannot precede the
			# buffer's start because, below, streamthinca will
			# be told the trigger list is complete upto this
			# buffer's time stamp.  this logic also requires
			# this method to be fed buffers in time order:  we
			# must never receive a buffer whose timestamp
			# precedes the timestamp of a buffer we have
			# already received.  NOTE:  the trigger objects'
			# comparison operators can be used for this test
			# directly, without explicitly retrieving .end
			assert all(event >= buf_timestamp for event in events)

			# assign IDs to triggers.  set all effective
			# distances to NaN.  gstlal_inspiral's effective
			# distances are incorrect, and the PE codes require
			# us to either provide correct effective distances
			# or communicate to them that they are incorrect.
			# they have explained that setting them to NaN is
			# sufficient for the latter.
			# FIXME:  fix the effective distances
			for event in events:
				event.process_id = self.coincs_document.process_id
				event.event_id = self.coincs_document.get_next_sngl_id()
				event.eff_distance = NaN

			# update likelihood snapshot if needed
			if self.likelihood_snapshot_interval is not None and (self.likelihood_snapshot_timestamp is None or buf_timestamp - self.likelihood_snapshot_timestamp >= self.likelihood_snapshot_interval):
				self.likelihood_snapshot_timestamp = buf_timestamp

				# post a checkpoint message.
				# FIXME:  make sure this triggers
				# self.snapshot_output_url() to be invoked.
				# lloidparts takes care of that for now,
				# but spreading the program logic around
				# like that isn't a good idea, this code
				# should be responsible for it somehow, no?
				# NOTE: self.snapshot_output_url() writes
				# the current rankingstat object to the
				# location identified by
				# .ranking_stat_output_url, so if that is
				# either not set or at least set to a
				# different name than
				# .ranking_stat_input_url the file that has
				# just been loaded above will not be
				# overwritten.
				self.pipeline.get_bus().post(message_new_checkpoint(self.pipeline, timestamp = buf_timestamp.ns()))

				# if a ranking statistic source url is set
				# and is not the same as the file to which
				# we are writing our ranking statistic data
				# then overwrite rankingstat with its
				# contents.  the use case is online
				# injection jobs that need to periodically
				# grab new ranking statistic data from
				# their corresponding non-injection partner
				if self.ranking_stat_input_url is not None and self.ranking_stat_input_url != self.ranking_stat_output_url:
					params_before = self.rankingstat.template_ids, self.rankingstat.instruments, self.rankingstat.min_instruments, self.rankingstat.delta_t
					self.rankingstat, _ = far.parse_likelihood_control_doc(ligolw_utils.load_url(self.ranking_stat_input_url, verbose = self.verbose, contenthandler = far.RankingStat.LIGOLWContentHandler))
					if params_before != (self.rankingstat.template_ids, self.rankingstat.instruments, self.rankingstat.min_instruments, self.rankingstat.delta_t):
						raise ValueError("'%s' contains incompatible ranking statistic configuration" % self.ranking_stat_input_url)

				# update streamthinca's ranking statistic
				# data
				if self.rankingstat.is_healthy(self.verbose):
					self.stream_thinca.ln_lr_from_triggers = far.OnlineFrankensteinRankingStat(self.rankingstat, self.rankingstat).finish().ln_lr_from_triggers
					if self.verbose:
						print("ranking statistic assignment ENABLED", file=sys.stderr)
				else:
					self.stream_thinca.ln_lr_from_triggers = None
					if self.verbose:
						print("ranking statistic assignment DISABLED", file=sys.stderr)

				# optionally get updated ranking statistic
				# PDF data and enable FAP/FAR assignment
				self.load_rankingstat_pdf()

			# add triggers to trigger rate record.  this needs
			# to be done without any cuts on coincidence, etc.,
			# so that the total trigger count agrees with the
			# total livetime from the SNR buffers.  we assume
			# the density of real signals is so small that this
			# count is not sensitive to their presence.  NOTE:
			# this is not true locally.  a genuine signal, if
			# loud, can significantly increase the local
			# density of triggers, therefore the trigger rate
			# measurement machinery must take care to average
			# the rate over a sufficiently long period of time
			# that the rate estimates are insensitive to the
			# presence of signals.  the current implementation
			# averages over whole science segments.  NOTE: this
			# must be done before running stream thinca (below)
			# so that the "how many instruments were on test"
			# is aware of this buffer.
			if not buf_is_gap:
				snr_min = self.rankingstat.snr_min
				for instrument in instruments:
					# FIXME At the moment, empty triggers are added to
					# inform the "how many instruments were on test", the
					# correct thing to do is probably to add metadata to
					# the buffer containing information about which
					# instruments were on
					self.rankingstat.denominator.triggerrates[instrument].add_ratebin(list(map(float, buf_seg[instrument])), len([event for event in events if event.snr >= snr_min and event.ifo == instrument]))

			# FIXME At the moment, empty triggers are added to
			# inform the "how many instruments were on test", the
			# correct thing to do is probably to add metadata to
			# the buffer containing information about which
			# instruments were on
			real_events = []
			for event in events:
				if event.snr >= snr_min:
					real_events.append(event)

			events = real_events

			# run stream thinca.
			instruments |= self.absent_instruments
			instruments |= self.rankingstat.instruments

			for instrument in instruments:
				if not self.stream_thinca.push(instrument, [event for event in events if event.ifo == instrument], buf_timestamp):
					continue

				flushed_sngls = self.stream_thinca.pull(self.rankingstat, fapfar = self.fapfar, zerolag_rankingstatpdf = self.zerolag_rankingstatpdf, coinc_sieve = self.rankingstat.fast_path_cut_from_triggers, cluster = self.cluster, cap_singles = self.cap_singles, FAR_trialsfactor = self.FAR_trialsfactor)
				self.coincs_document.commit()

				# do GraceDB alerts and update eye candy
				self.__do_gracedb_alerts(self.stream_thinca.last_coincs)
				self.eye_candy.update(events, self.stream_thinca.last_coincs)

				# after doing alerts, no longer need
				# per-trigger SNR data for the triggers
				# that are too old to be used to form
				# candidates.
				for event in flushed_sngls:
					del event.H1_snr_time_series
					del event.L1_snr_time_series
					del event.V1_snr_time_series
					del event.K1_snr_time_series


	def _record_horizon_distance(self, instrument, timestamp, horizon_distance):
		"""
		timestamp can be a float or a slice with float boundaries.
		"""
		# retrieve the horizon history for this instrument
		horizon_history = self.rankingstat.numerator.horizon_history[instrument]
		# NOTE:  timestamps are floats here (or float slices), not
		# LIGOTimeGPS.  whitener should be reporting PSDs with
		# integer timestamps so the timestamps are not naturally
		# high precision, and, anyway, we don't need nanosecond
		# precision for the horizon distance history.
		horizon_history[timestamp] = horizon_distance


	def record_horizon_distance(self, instrument, timestamp, horizon_distance):
		"""
		timestamp can be a float or a slice with float boundaries.
		"""
		with self.lock:
			self._record_horizon_distance(instrument, timestamp, horizon_distance)


	def horizgatehandler(self, elem, timestamp, instrument_tpl):
		"""!
		A handler that intercepts h(t) gate state transitions to 0
		horizon distances.

		@param elem A reference to the lal_gate element or None
		(only used for verbosity)
		@param timestamp A gstreamer time stamp that marks the
		state transition (in nanoseconds)
		@param instrument the instrument this state transtion is to
		be attributed to, e.g., "H1", etc..
		@param new_state the state transition, must be either True
		or False
		"""
		# essentially we want to set the horizon distance record to
		# 0 at both on-to-off and off-to-on transitions so that the
		# horizon distance is reported to be 0 at all times within
		# the "off" period.  the problem is gaps due to glitch
		# excision, which is done using a gate that comes after the
		# whitener.  the whitener sees data flowing during these
		# periods and can report PSDs (and thus trigger horizon
		# distances to get recorded) but because of the glitch
		# excision the real sensitivity of the instrument is 0.
		# there's not much we can do to prevent the PSD from being
		# reported, but we try to retroactively correct the problem
		# when it occurs by using a slice to re-zero the entire
		# "off" interval preceding each off-to-on transition.  this
		# requires access to the segment data to get the timestamp
		# for the start of the off interval, but we could also
		# record that number in this class here for our own use if
		# the entanglement with the segments tracker becomes a
		# problem.
		#
		# FIXME:  there is a potential race condition in that we're
		# relying on all spurious PSD messages that we are to
		# override with 0 to already have been reported when this
		# method gets invoked by the whitehtsegments, i.e., we're
		# relying on the whitehtsegments gate to report state
		# transitions *after* any potential whitener PSD messages
		# have been generated.  this is normally guaranteed because
		# the whitener does not generate PSD messages during gap
		# intervals.

		timestamp = float(LIGOTimeGPS(0, timestamp))
		instrument, new_state = instrument_tpl

		if self.verbose:
			print("%s: %s horizon distance state transition: %s @ %s" % (elem.get_name(), instrument, ("on" if new_state else "off"), str(timestamp)), file=sys.stderr)

		with self.lock:
			# retrieve the horizon history for this instrument
			horizon_history = self.rankingstat.numerator.horizon_history[instrument]

			if new_state:
				# this if statement is checking if
				# ~self.seglistdicts[segtype][instrument]
				# is empty, and if not then it zeros the
				# interval spanned by the last segment in
				# that list, but what's done below avoids
				# the explicit segment arithmetic
				segtype = "whitehtsegments"
				if len(self.segmentstracker.seglistdicts[segtype][instrument]) > 1:
					self._record_horizon_distance(instrument, slice(float(self.segmentstracker.seglistdicts[segtype][instrument][-2][-1]), timestamp), 0.)
				else:
					self._record_horizon_distance(instrument, timestamp, 0.)
			elif self.rankingstat.numerator.horizon_history[instrument]:
				self._record_horizon_distance(instrument, slice(timestamp, None), 0.)
			else:
				self._record_horizon_distance(instrument, timestamp, 0.)


	def checkpoint(self, timestamp):
		"""!
		Checkpoint, e.g., flush segments and triggers to disk.

		@param timestamp the LIGOTimeGPS timestamp of the current
		buffer in order to close off open segment intervals before
		writing to disk
		"""
		try:
			self.snapshot_output_url("%s_LLOID" % self.tag, "xml.gz", verbose = self.verbose)
		except TypeError as te:
			print("Warning: couldn't build output file on checkpoint, probably there aren't any triggers: %s" % te, file=sys.stderr)
		# FIXME:  the timestamp is used to close off open segments
		# and so should *not* be the timestamp of the current
		# buffer, necessarily, but rather a GPS time guaranteed to
		# precede any future state transitions of any segment list.
		# especially if the pipeline is ever run in an "advanced
		# warning" configuration using the GPS time of a trigger
		# buffer would be an especially bad choice.
		self.segmentstracker.flush_segments_to_disk(self.tag, timestamp)


	def __get_psd_xmldoc(self):
		xmldoc = lal.series.make_psd_xmldoc(self.psds)
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_inspiral", {}, ifos = self.psds)
		ligolw_process.set_process_end_time(process)
		return xmldoc


	def web_get_psd_xml(self):
		with self.lock:
			output = io.StringIO()
			ligolw_utils.write_fileobj(self.__get_psd_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr


	def __get_rankingstat_xmldoc(self, clipped = False):
		# generate a ranking statistic output document.  NOTE:  if
		# we are in possession of ranking statistic PDFs then those
		# are included in the output.  this allows a single
		# document to be uploaded to gracedb.  in an online
		# analysis, those PDFs come from the marginlization process
		# and represent the full distribution of ranking statistics
		# across the search, and include with them the analysis'
		# total observed zero-lag ranking statistic histogram ---
		# everything required to re-evaluate the FAP and FAR for an
		# uploaded candidate.
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, u"gstlal_inspiral", paramdict = {}, ifos = self.rankingstat.instruments)
		# FIXME:  don't do this.  find a way to reduce the storage
		# requirements of the horizon distance history and then go
		# back to uploading the full file to gracedb
		if clipped:
			rankingstat = self.rankingstat.copy()
			try:
				endtime = rankingstat.numerator.horizon_history.maxkey()
			except ValueError:
				# empty horizon history
				pass
			else:
				# keep the last hour of history
				endtime -= 3600. * 1
				for history in rankingstat.numerator.horizon_history.values():
					del history[:endtime]
		else:
			rankingstat = self.rankingstat
		far.gen_likelihood_control_doc(xmldoc, rankingstat, self.rankingstatpdf)
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def __get_p_astro_json(self, lr, m1, m2, snr, far):
		return p_astro_gstlal.compute_p_astro(lr, m1, m2, snr, far, self.rankingstatpdf.copy(), activation_counts = self.activation_counts)

	def __get_rankingstat_xmldoc_for_gracedb(self):
		# FIXME:  remove this wrapper when the horizon history
		# encoding is replaced with something that uses less space
		return self.__get_rankingstat_xmldoc(clipped = True)


	def web_get_rankingstat(self):
		with self.lock:
			output = io.StringIO()
			ligolw_utils.write_fileobj(self.__get_rankingstat_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
			return outstr


	def __get_zerolag_rankingstatpdf_xmldoc(self):
		xmldoc = ligolw.Document()
		xmldoc.appendChild(ligolw.LIGO_LW())
		process = ligolw_process.register_to_xmldoc(xmldoc, u"gstlal_inspiral", paramdict = {}, ifos = self.rankingstat.instruments)
		far.gen_likelihood_control_doc(xmldoc, None, self.zerolag_rankingstatpdf)
		ligolw_process.set_process_end_time(process)
		return xmldoc


	def web_get_zerolag_rankingstatpdf(self):
		with self.lock:
			output = io.StringIO()
			ligolw_utils.write_fileobj(self.__get_zerolag_rankingstatpdf_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
			return outstr


	def __flush(self):
		# run StreamThinca in flush mode.  forms candidates from
		# whatever triggers remain in the queues, and processes
		# them

		flushed_sngls = self.stream_thinca.pull(self.rankingstat, fapfar = self.fapfar, zerolag_rankingstatpdf = self.zerolag_rankingstatpdf, coinc_sieve = self.rankingstat.fast_path_cut_from_triggers, flush = True, cluster = self.cluster, cap_singles = self.cap_singles, FAR_trialsfactor = self.FAR_trialsfactor)
		self.coincs_document.commit()

		# do GraceDB alerts
		self.__do_gracedb_alerts(self.stream_thinca.last_coincs)

		# after doing alerts, no longer need per-trigger SNR data
		# for the triggers that are too old to be used to form
		# candidates.
		for event in flushed_sngls:
			del event.H1_snr_time_series
			del event.L1_snr_time_series
			del event.V1_snr_time_series
			del event.K1_snr_time_series


	def __do_gracedb_alerts(self, last_coincs):
		# check for no-op
		if self.rankingstatpdf is None or not self.rankingstatpdf.is_healthy():
			return
		# sanity check.  all code paths that result in an
		# initialized and healthy rankingstatpdf object also
		# initialize a fapfar object from it;  this is here to make
		# sure that remains true
		assert self.fapfar is not None

		# do alerts
		self.gracedbwrapper.do_alerts(last_coincs, self.psds, self.__get_rankingstat_xmldoc_for_gracedb, self.segmentstracker.seglistdicts, self.__get_p_astro_json)


	def web_get_sngls_snr_threshold(self):
		with self.lock:
			if self.stream_thinca.sngls_snr_threshold is not None:
				yield "snr=%.17g\n" % self.stream_thinca.sngls_snr_threshold
			else:
				yield "snr=\n"


	def web_set_sngls_snr_threshold(self):
		try:
			with self.lock:
				snr_threshold = bottle.request.forms["snr"]
				if snr_threshold:
					self.stream_thinca.sngls_snr_threshold = float(snr_threshold)
					yield "OK: snr=%.17g\n" % self.stream_thinca.sngls_snr_threshold
				else:
					self.stream_thinca.sngls_snr_threshold = None
					yield "OK: snr=\n"
		except:
			yield "error\n"


	def __write_output_url(self, url = None, verbose = False):
		self.__flush()
		if url is not None:
			self.coincs_document.url = url
		with self.segmentstracker.lock:
			self.coincs_document.write_output_url(seglistdicts = self.segmentstracker.seglistdicts, verbose = verbose)
		# can't be used anymore
		del self.coincs_document


	def __write_ranking_stat_url(self, url, description, snapshot = False, verbose = False):
		# write the ranking statistic file.
		ligolw_utils.write_url(self.__get_rankingstat_xmldoc(), url, gz = (url or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)
		# Snapshots get their own custom file and path
		if snapshot:
			fname = self.segmentstracker.T050017_filename(description + '_DISTSTATS', 'xml.gz')
			shutil.copy(ligolw_utils.local_path_from_url(url), os.path.join(subdir_from_T050017_filename(fname), fname))


	def write_output_url(self, url = None, description = "", verbose = False):
		with self.lock:
			if self.ranking_stat_output_url is not None:
				self.__write_ranking_stat_url(self.ranking_stat_output_url, description, verbose = verbose)
			self.__write_output_url(url = url, verbose = verbose)


	def snapshot_output_url(self, description, extension, verbose = False):
		with self.lock:
			fname = self.segmentstracker.T050017_filename(description, extension)
			fname = os.path.join(subdir_from_T050017_filename(fname), fname)
			if self.ranking_stat_output_url is not None:
				self.__write_ranking_stat_url(self.ranking_stat_output_url, description, snapshot = True, verbose = verbose)
			if self.zerolag_rankingstatpdf_url is not None:
				ligolw_utils.write_url(self.__get_zerolag_rankingstatpdf_xmldoc(), self.zerolag_rankingstatpdf_url, gz = (self.zerolag_rankingstatpdf_url or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)
			coincs_document = self.coincs_document.get_another()
			self.__write_output_url(url = fname, verbose = verbose)
			self.coincs_document = coincs_document
			# NOTE:  this operation requires stream_thinca to
			# have been flushed by calling .pull() with flush =
			# True.  the .__write_output_url() code path does
			# that, but there is no check here to ensure that
			# remains true so be careful if you are editing
			# these methods.
			self.stream_thinca.set_xmldoc(self.coincs_document.xmldoc, self.coincs_document.process_id)
