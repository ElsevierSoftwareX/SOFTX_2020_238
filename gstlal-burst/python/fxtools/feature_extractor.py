# Copyright (C) 2017-2018  Sydney J. Chamberlin, Patrick Godwin, Chad Hanna, Duncan Meacher
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

## @file

## @package feature_extractor


# =============================
# 
#           preamble
#
# =============================


from collections import deque
import itertools
import json
import optparse
import os
import StringIO
import threading
import shutil

import h5py
import numpy

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from ligo import segments
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import process as ligolw_process

from gstlal import aggregator
from gstlal import bottle
from gstlal import pipeio
from gstlal import pipeparts
from gstlal import simplehandler

from gstlal.fxtools import sngltriggertable
from gstlal.fxtools import utils


# =============================
# 
#           classes
#
# =============================


class MultiChannelHandler(simplehandler.Handler):
	"""
	A subclass of simplehandler.Handler to be used with 
	multiple channels.

	Implements additional message handling for dealing with spectrum
	messages and creates trigger files containing features for use in iDQ.
	"""
	def __init__(self, mainloop, pipeline, logger, data_source_info, options, **kwargs):
		self.lock = threading.Lock()
		self.logger = logger
		self.out_path = options.out_path
		self.instrument = data_source_info.instrument
		self.frame_segments = data_source_info.frame_segments
		self.keys = kwargs.pop("keys")
		self.num_samples = len(self.keys)
		self.sample_rate = options.sample_rate
		self.waveforms = kwargs.pop("waveforms")
		self.basename = kwargs.pop("basename")
		self.waveform_type = options.waveform

		# format keys used for saving, etc.
		self.aggregate_rate = True # NOTE: hard-coded for now
		if self.aggregate_rate:
			self.keys = list(set([key[0] for key in self.keys]))
		else:
			self.keys = [os.path.join(key[0], str(key[1]).zfill(4)) for key in self.keys]

		# format id for aesthetics
		self.job_id = str(options.job_id).zfill(4)
		self.subset_id = str(kwargs.pop("subset_id")).zfill(4)

		### iDQ saving properties
		self.timestamp = None
		self.last_save_time = None
		self.last_persist_time = None
		self.cadence = options.cadence
		self.persist_cadence = options.persist_cadence
		self.feature_start_time = options.feature_start_time
		self.feature_end_time = options.feature_end_time
		self.columns = ['start_time', 'stop_time', 'trigger_time', 'frequency', 'q', 'snr', 'phase', 'sigmasq', 'chisq']

		### set up queue to cache features depending on pipeline mode
		self.feature_mode = options.feature_mode
		if self.feature_mode == 'timeseries':
			self.feature_queue = utils.TimeseriesFeatureQueue(self.keys, self.columns, sample_rate = self.sample_rate)
		elif self.feature_mode == 'etg':
			self.feature_queue = utils.ETGFeatureQueue(self.keys, self.columns)

		# set whether data source is live
		self.is_live = data_source_info.data_source in data_source_info.live_sources

		# get base temp directory
		if '_CONDOR_SCRATCH_DIR' in os.environ:
			self.tmp_dir = os.environ['_CONDOR_SCRATCH_DIR']
		else:
			self.tmp_dir = os.environ['TMPDIR']

		# feature saving properties
		self.save_format = options.save_format
		if self.save_format == 'hdf5':
			if self.feature_mode == 'timeseries':
				self.fdata = utils.HDF5TimeseriesFeatureData(self.columns, keys = self.keys, cadence = self.cadence, sample_rate = self.sample_rate)
			elif self.feature_mode == 'etg':
				self.fdata = utils.HDF5ETGFeatureData(self.columns, keys = self.keys, cadence = self.cadence)
			else:
				raise KeyError, 'not a valid feature mode option'

		elif self.save_format == 'ascii':
			self.header = "# %18s\t%20s\t%20s\t%10s\t%8s\t%8s\t%8s\t%10s\t%s\n" % ("start_time", "stop_time", "trigger_time", "frequency", "phase", "q", "chisq", "snr", "channel")
			self.fdata = deque(maxlen = 25000)
			self.fdata.append(self.header)

		elif self.save_format == 'kafka':
			self.data_transfer = options.data_transfer
			self.kafka_partition = options.kafka_partition
			self.kafka_topic = options.kafka_topic
			self.kafka_conf = {'bootstrap.servers': options.kafka_server}
			self.producer = Producer(self.kafka_conf)

		elif self.save_format == 'bottle':
			assert not options.disable_web_service, 'web service is not available to use bottle to transfer features'
			self.feature_data = deque(maxlen = 2000)
			bottle.route("/feature_subset")(self.web_get_feature_data)

		# set up bottle routes for PSDs
		self.psds = {}
		if not options.disable_web_service:
			bottle.route("/psds.xml")(self.web_get_psd_xml)

		super(MultiChannelHandler, self).__init__(mainloop, pipeline, **kwargs)

	def do_on_message(self, bus, message):
		"""!
		Handle application-specific message types, 
		e.g., spectrum messages.
		
		@param bus: A reference to the pipeline's bus
		@param message: A reference to the incoming message
		"""
		#
		# return value of True tells parent class that we have done
		# all that is needed in response to the message, and that
		# it should ignore it.  a return value of False means the
		# parent class should do what it thinks should be done
		#
		if message.type == Gst.MessageType.ELEMENT:
			if message.get_structure().get_name() == "spectrum":
				# get the channel name & psd.
				instrument, info = message.src.get_name().split("_", 1)
				channel, _ = info.rsplit("_", 1)
				psd = pipeio.parse_spectrum_message(message)
				# save psd
				self.psds[channel] = psd
				return True		
		return False

	def bufhandler(self, elem, sink_dict):
		"""
		Processes rows from a Gstreamer buffer and
		handles conditions for file saving.

		@param elem: A reference to the gstreamer element being processed
		@param sink_dict: A dictionary containing references to gstreamer elements
		"""
		with self.lock:
			buf = elem.emit("pull-sample").get_buffer()
			buftime = int(buf.pts / 1e9)
			channel, rate  = sink_dict[elem]

			# push new stream event to queue if done processing current timestamp
			if len(self.feature_queue):
				feature_subset = self.feature_queue.pop()
				self.timestamp = feature_subset['timestamp']

				# set save times and initialize specific saving properties if not already set
				if self.last_save_time is None:
					self.last_save_time = self.timestamp
					self.last_persist_time = self.timestamp
					if self.save_format =='hdf5':
						duration = utils.floor_div(self.timestamp + self.persist_cadence, self.persist_cadence) - self.timestamp
						self.set_hdf_file_properties(self.timestamp, duration)

				# Save triggers once per cadence if saving to disk
				if self.save_format == 'hdf5' or self.save_format == 'ascii':
					if self.timestamp and utils.in_new_epoch(self.timestamp, self.last_save_time, self.cadence) or (self.timestamp == self.feature_end_time):
						self.logger.info("saving features to disk at timestamp = %d" % self.timestamp)
						if self.save_format == 'hdf5':
							self.to_hdf_file()
						elif self.save_format == 'ascii':
							self.to_trigger_file(self.timestamp)
							self.fdata.clear()
							self.fdata.append(self.header)
						self.last_save_time = self.timestamp

				# persist triggers once per persist cadence if using hdf5 format
				if self.save_format == 'hdf5':
					if self.timestamp and utils.in_new_epoch(self.timestamp, self.last_persist_time, self.persist_cadence):
						self.logger.info("persisting features to disk at timestamp = %d" % self.timestamp)
						self.finish_hdf_file()
						self.last_persist_time = self.timestamp
						self.set_hdf_file_properties(self.timestamp, self.persist_cadence)

				# add features to respective format specified
				if self.save_format == 'kafka':
					if self.data_transfer == 'table':
						self.producer.produce(timestamp = self.timestamp, topic = self.kafka_topic, value = json.dumps(feature_subset))
					elif self.data_transfer == 'row':
						for row in itertools.chain(*feature_subset['features'].values()):
							if row:
								self.producer.produce(timestamp = self.timestamp, topic = self.kafka_topic, value = json.dumps(row))
					self.producer.poll(0) ### flush out queue of sent packets
				elif self.save_format == 'bottle':
					self.feature_data.append(feature_subset)
				elif self.save_format == 'hdf5':
					self.fdata.append(self.timestamp, feature_subset['features'])

			# read buffer contents
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
				if mapinfo.data:
					if (buftime >= self.feature_start_time and buftime <= self.feature_end_time):
						for row in sngltriggertable.GSTLALSnglTrigger.from_buffer(mapinfo.data):
							self.process_row(channel, rate, buftime, row)
				memory.unmap(mapinfo)

			del buf
			return Gst.FlowReturn.OK

	def process_row(self, channel, rate, buftime, row):
		"""
		Given a channel, rate, and the current buffer
		time, will process a row from a gstreamer buffer.
		"""
		# if segments provided, ensure that trigger falls within these segments
		if self.frame_segments[self.instrument]:
			trigger_seg = segments.segment(LIGOTimeGPS(row.end_time, row.end_time_ns), LIGOTimeGPS(row.end_time, row.end_time_ns))

		if not self.frame_segments[self.instrument] or self.frame_segments[self.instrument].intersects_segment(trigger_seg):
			freq, q, duration = self.waveforms[channel].parameter_grid[rate][row.channel_index]
			filter_duration = self.waveforms[channel].filter_duration[rate]
			filter_stop_time = row.end_time + row.end_time_ns * 1e-9

			# set trigger time based on waveform
			if self.waveform_type == 'sine_gaussian':
				trigger_time = filter_stop_time
				start_time = trigger_time - duration / 2.
				stop_time = trigger_time + duration / 2.

			elif self.waveform_type == 'half_sine_gaussian':
				trigger_time = filter_stop_time
				start_time = trigger_time - duration
				stop_time = trigger_time

			# append row for data transfer/saving
			timestamp = int(numpy.floor(trigger_time))
			feature_row = {'timestamp':timestamp, 'channel':channel, 'start_time':start_time, 'stop_time':stop_time, 'snr':row.snr,
			               'trigger_time':trigger_time, 'frequency':freq, 'q':q, 'phase':row.phase, 'sigmasq':row.sigmasq, 'chisq':row.chisq}
			self.feature_queue.append(timestamp, channel, feature_row)

			# save iDQ compatible data
			if self.save_format == 'ascii':
				channel_tag = ('%s_%i_%i' %(channel, rate/4, rate/2)).replace(":","_",1)
				self.fdata.append("%20.9f\t%20.9f\t%20.9f\t%10.3f\t%8.3f\t%8.3f\t%8.3f\t%10.3f\t%s\n" % (start_time, stop_time, trigger_time, freq, row.phase, q, row.chisq, row.snr, channel_tag))


	def to_trigger_file(self, buftime = None):
		"""
		Dumps triggers saved in memory to disk, following an iDQ ingestion format.
		Contains a header specifying aligned columns, along with triggers, one per row.
		Uses the T050017 filenaming convention.
		NOTE: This method should only be called by an instance that is locked.
		"""
		# Only write triggers to disk where the associated data structure has more
		# than the header stored within.
		if len(self.fdata) > 1 :
			fname = '%s-%d-%d.%s' % (self.tag, utils.floor_div(self.last_save_time, self.cadence), self.cadence, "trg")
			path = os.path.join(self.out_path, self.tag, self.tag+"-"+str(fname.split("-")[2])[:5])
			fpath = os.path.join(path, fname)
			tmpfile = fpath+"~"
			try:
				os.makedirs(path)
			except OSError:
				pass
			with open(tmpfile, 'w') as f:
 				f.write(''.join(self.fdata))
			shutil.move(tmpfile, fpath)
			if buftime:
				latency = numpy.round(int(aggregator.now()) - buftime)
				self.logger.info("buffer timestamp = %d, latency at write stage = %d" % (buftime, latency))

	def to_hdf_file(self):
		"""
		Dumps triggers saved in memory to disk in hdf5 format.
		Uses the T050017 filenaming convention.
		NOTE: This method should only be called by an instance that is locked.
		"""
		self.fdata.dump(self.tmp_path, self.fname, utils.floor_div(self.last_save_time, self.cadence), tmp = True)

	def finish_hdf_file(self):
		"""
		Move a temporary hdf5 file to its final location after
		all file writes have been completed.
		"""
		final_path = os.path.join(self.fpath, self.fname)+".h5"
		tmp_path = os.path.join(self.tmp_path, self.fname)+".h5.tmp"
		shutil.move(tmp_path, final_path)

	def finalize(self):
		"""
		Clears out remaining features from the queue for saving to disk.
		"""
		# save remaining triggers
		if self.save_format == 'hdf5':
			self.feature_queue.flush()
			while len(self.feature_queue):
				feature_subset = self.feature_queue.pop()
				self.fdata.append(feature_subset['timestamp'], feature_subset['features'])

			self.to_hdf_file()
			self.finish_hdf_file()

		elif self.save_format == 'ascii':
			self.to_trigger_file()

	def set_hdf_file_properties(self, start_time, duration):
		"""
		Returns the file name, as well as locations of temporary and permanent locations of
		directories where triggers will live, when given the current gps time and a gps duration.
		Also takes care of creating new directories as needed and removing any leftover temporary files.
		"""
		# set/update file names and directories with new gps time and duration
		self.fname = os.path.splitext(utils.to_trigger_filename(self.basename, start_time, duration, 'h5'))[0]
		self.fpath = utils.to_trigger_path(os.path.abspath(self.out_path), self.basename, start_time, self.job_id, self.subset_id)
		self.tmp_path = utils.to_trigger_path(self.tmp_dir, self.basename, start_time, self.job_id, self.subset_id)

		# create temp and output directories if they don't exist
		aggregator.makedir(self.fpath)
		aggregator.makedir(self.tmp_path)

		# delete leftover temporary files
		tmp_file = os.path.join(self.tmp_path, self.fname)+'.h5.tmp'
		if os.path.isfile(tmp_file):
			os.remove(tmp_file)

	def gen_psd_xmldoc(self):
		xmldoc = lal.series.make_psd_xmldoc(self.psds)
		process = ligolw_process.register_to_xmldoc(xmldoc, "gstlal_idq", {})
		ligolw_process.set_process_end_time(process)
		return xmldoc

	def web_get_psd_xml(self):
		with self.lock:
			output = StringIO.StringIO()
			ligolw_utils.write_fileobj(self.gen_psd_xmldoc(), output)
			outstr = output.getvalue()
			output.close()
		return outstr

	def web_get_feature_data(self):
		header = {'Content-type': 'application/json'}
		# if queue is empty, send appropriate response
		if not self.feature_data:
			status = 204
			body = json.dumps({'error': "No Content"})
		# else, get feature data and send as JSON
		else:
			status = 200
			with self.lock:
				body = json.dumps(self.feature_data.popleft())
		return bottle.HTTPResponse(status = status, headers = header, body = body)

class LinkedAppSync(pipeparts.AppSync):
	def __init__(self, appsink_new_buffer, sink_dict = None):
		self.time_ordering = 'full'
		if sink_dict:
			self.sink_dict = sink_dict
		else:
			self.sink_dict = {}
		super(LinkedAppSync, self).__init__(appsink_new_buffer, self.sink_dict.keys())
	
	def attach(self, appsink):
		"""
		connect this AppSync's signal handlers to the given appsink
		element.  the element's max-buffers property will be set to
		1 (required for AppSync to work).
		"""
		if appsink in self.appsinks:
			raise ValueError("duplicate appsinks %s" % repr(appsink))
		appsink.set_property("max-buffers", 1)
		handler_id = appsink.connect("new-preroll", self.new_preroll_handler)
		assert handler_id > 0
		handler_id = appsink.connect("new-sample", self.new_sample_handler)
		assert handler_id > 0
		handler_id = appsink.connect("eos", self.eos_handler)
		assert handler_id > 0
		self.appsinks[appsink] = None
		_, rate, channel = appsink.name.split("_", 2)
		self.sink_dict.setdefault(appsink, (channel, int(rate)))
		return appsink
	
	def pull_buffers(self, elem):
		"""
		for internal use.  must be called with lock held.
		"""

		while 1:
			if self.time_ordering == 'full':
				# retrieve the timestamps of all elements that
				# aren't at eos and all elements at eos that still
				# have buffers in them
				timestamps = [(t, e) for e, t in self.appsinks.iteritems() if e not in self.at_eos or t is not None]
				# if all elements are at eos and none have buffers,
				# then we're at eos
				if not timestamps:
					return Gst.FlowReturn.EOS
				# find the element with the oldest timestamp.  None
				# compares as less than everything, so we'll find
				# any element (that isn't at eos) that doesn't yet
				# have a buffer (elements at eos and that are
				# without buffers aren't in the list)
				timestamp, elem_with_oldest = min(timestamps)
				# if there's an element without a buffer, quit for
				# now --- we require all non-eos elements to have
				# buffers before proceding
				if timestamp is None:
					return Gst.FlowReturn.OK
				# clear timestamp and pass element to handler func.
				# function call is done last so that all of our
				# book-keeping has been taken care of in case an
				# exception gets raised
				self.appsinks[elem_with_oldest] = None
				self.appsink_new_buffer(elem_with_oldest, self.sink_dict)
	
			elif self.time_ordering == 'partial':
				# retrieve the timestamps of elements of a given channel
				# that aren't at eos and all elements at eos that still
				# have buffers in them
				channel = self.sink_dict[elem][0]
				timestamps = [(t, e) for e, t in self.appsinks.iteritems() if self.sink_dict[e][0] == channel and (e not in self.at_eos or t is not None)]
				# if all elements are at eos and none have buffers,
				# then we're at eos
				if not timestamps:
					return Gst.FlowReturn.EOS
				# find the element with the oldest timestamp.  None
				# compares as less than everything, so we'll find
				# any element (that isn't at eos) that doesn't yet
				# have a buffer (elements at eos and that are
				# without buffers aren't in the list)
				timestamp, elem_with_oldest = min(timestamps)
				# if there's an element without a buffer, quit for
				# now --- we require all non-eos elements to have
				# buffers before proceding
				if timestamp is None:
					return Gst.FlowReturn.OK
				# clear timestamp and pass element to handler func.
				# function call is done last so that all of our
				# book-keeping has been taken care of in case an
				# exception gets raised
				self.appsinks[elem_with_oldest] = None
				self.appsink_new_buffer(elem_with_oldest, self.sink_dict)
			
			elif self.time_ordering == 'none':
				if not elem in self.appsinks:
					return Gst.FlowReturn.EOS
				if self.appsinks[elem] is None:
					return Gst.FlowReturn.OK
				self.appsinks[elem] = None
				self.appsink_new_buffer(elem, self.sink_dict)


# =============================
# 
#           utilities
#
# =============================

def append_options(parser):
	"""!
	Append feature extractor options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout executables
	that use feature extractor based utilities.
	"""
	group = optparse.OptionGroup(parser, "Waveform Options", "Adjust waveforms/parameter space used for feature extraction")
	group.add_option("-m", "--mismatch", type = "float", default = 0.2, help = "Mismatch between templates, mismatch = 1 - minimal match. Default = 0.2.")
	group.add_option("-q", "--qhigh", type = "float", default = 20, help = "Q high value for half sine-gaussian waveforms. Default = 20.")
	group.add_option("--waveform", metavar = "string", default = "half_sine_gaussian", help = "Specifies the waveform used for matched filtering. Possible options: (half_sine_gaussian, sine_gaussian). Default = half_sine_gaussian")
	parser.add_option_group(group)

	group = optparse.OptionGroup(parser, "Saving Options", "Adjust parameters used for saving/persisting features to disk as well as directories specified")
	group.add_option("--out-path", metavar = "path", default = ".", help = "Write to this path. Default = .")
	group.add_option("--description", metavar = "string", default = "GSTLAL_IDQ_FEATURES", help = "Set the filename description in which to save the output.")
	group.add_option("--save-format", metavar = "string", default = "hdf5", help = "Specifies the save format (ascii/hdf5/kafka/bottle) of features written to disk. Default = hdf5")
	group.add_option("--feature-mode", metavar = "string", default = "timeseries", help = "Specifies the mode for which features are generated (timeseries/etg). Default = timeseries")
	group.add_option("--data-transfer", metavar = "string", default = "table", help = "Specifies the format of features transferred over-the-wire (table/row). Default = table")
	group.add_option("--cadence", type = "int", default = 20, help = "Rate at which to write trigger files to disk. Default = 20 seconds.")
	group.add_option("--persist-cadence", type = "int", default = 200, help = "Rate at which to persist trigger files to disk, used with hdf5 files. Needs to be a multiple of save cadence. Default = 200 seconds.")
	parser.add_option_group(group)

	group = optparse.OptionGroup(parser, "Kafka Options", "Adjust settings used for pushing extracted features to a Kafka topic.")
	group.add_option("--kafka-partition", metavar = "string", help = "If using Kafka, sets the partition that this feature extractor is assigned to.")
	group.add_option("--kafka-topic", metavar = "string", help = "If using Kafka, sets the topic name that this feature extractor publishes feature vector subsets to.")
	group.add_option("--kafka-server", metavar = "string", help = "If using Kafka, sets the server url that the kafka topic is hosted on.")
	group.add_option("--job-id", type = "string", default = "0001", help = "Sets the job identication of the feature extractor with a 4 digit integer string code, padded with zeros. Default = 0001")
	parser.add_option_group(group)

	group = optparse.OptionGroup(parser, "Program Behavior")
	group.add_option("--local-frame-caching", action = "store_true", help = "Pre-reads frame data and stores to local filespace.")
	group.add_option("--disable-web-service", action = "store_true", help = "If set, disables web service that allows monitoring of PSDS of aux channels.")
	group.add_option("-v", "--verbose", action = "store_true", help = "Be verbose.")
	group.add_option("--nxydump-segment", metavar = "start:stop", help = "Set the time interval to dump from nxydump elements (optional).")
	group.add_option("--sample-rate", type = "int", metavar = "Hz", default = 1, help = "Set the sample rate for feature timeseries output, must be a power of 2. Default = 1 Hz.")
	group.add_option("--snr-threshold", type = "float", default = 5.5, help = "Specifies the SNR threshold for features written to disk, required if 'feature-mode' option is set. Default = 5.5")
	group.add_option("--feature-start-time", type = "int", metavar = "seconds", help = "Set the start time of the segment to output features in GPS seconds. Required unless --data-source=lvshm")
	group.add_option("--feature-end-time", type = "int", metavar = "seconds", help = "Set the end time of the segment to output features in GPS seconds.  Required unless --data-source=lvshm")
	parser.add_option_group(group)
