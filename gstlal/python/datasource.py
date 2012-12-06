# Copyright (C) 2009--2011  Kipp Cannon, Chad Hanna, Drew Keppel
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import sys
import optparse
import math

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from gstlal import bottle
from gstlal import pipeparts
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw import utils
from glue.ligolw import ligolw
from glue import segments
from pylal.datatypes import LIGOTimeGPS


#
# Misc useful functions
#


def channel_dict_from_channel_list(channel_list):
	"""
	Given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""

	channel_dict = {}
	for channel in channel_list:
		ifo = channel.split("=")[0]
		chan = "".join(channel.split("=")[1:])
		channel_dict[ifo] = chan

	return channel_dict


def pipeline_channel_list_from_channel_dict(channel_dict, ifos = None, opt = "channel-name"):
	"""
	Produce a string of channel name arguments suitable for a pipeline.py
	program that doesn't technically allow multiple options. For example
	--channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN
	"""

	outstr = ""
	if ifos is None:
		ifos = channel_dict.keys()
	for i, ifo in enumerate(ifos):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[ifo])
		else:
			outstr += "--%s=%s=%s " % (opt, ifo, channel_dict[ifo])

	return outstr


def state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list, state_vector_on_off_dict = {"H1" : [0x7, 0x160], "H2" : [0x7, 0x160], "L1" : [0x7, 0x160], "V1" : [0x67, 0x100]}):
	"""
	Produce a dictionary (keyed by detector) of on / off bit tuples from a
	list provided on the command line.
	"""

	for line in on_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		try:
			state_vector_on_off_dict[ifo][0] = int(bits)
		except ValueError: # must be hex
			state_vector_on_off_dict[ifo][0] = int(bits, 16)
	
	for line in off_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		try:
			state_vector_on_off_dict[ifo][1] = int(bits)
		except ValueError: # must be hex
			state_vector_on_off_dict[ifo][1] = int(bits, 16)

	return state_vector_on_off_dict


def state_vector_on_off_list_from_bits_dict(bit_dict):
	"""
	Produce a commandline useful list from a dictionary of on / off state
	vector bits keyed by detector.
	"""

	onstr = ""
	offstr = ""
	for i, ifo in enumerate(bit_dict):
		if i == 0:
			onstr += "%s=%s " % (ifo, bit_dict[ifo][0])
			offstr += "%s=%s " % (ifo, bit_dict[ifo][1])
		else:
			onstr += "--state-vector-on-bits=%s=%s " % (ifo, bit_dict[ifo][0])
			offstr += "--state-vector-off-bits=%s=%s " % (ifo, bit_dict[ifo][1])

	return onstr, offstr


#
# Class to hold the data associated with data sources
#


class GWDataSourceInfo(object):

	def __init__(self, options):
		data_sources = ("frames", "online", "nds", "white", "silence", "AdvVirgo", "LIGO", "AdvLIGO")

		# Callbacks to handle the "start" and "stop" signals from the gate
		# element. This is useful for doing things like segments
		self.gate_start_callback = None
		self.gate_stop_callback = None

		# Sanity check the options
		if options.data_source not in data_sources:
			raise ValueError("--data-source not in " + repr(data_sources))
		if options.data_source == "frames" and options.frame_cache is None:
			raise ValueError("--frame-cache must be specified when using --data-source=frames")
		if (options.gps_start_time is None or options.gps_end_time is None) and options.data_source == "frames":
			raise ValueError("--gps-start-time and --gps-end-time must be specified unless --data-source=online")
		if len(options.channel_name) == 0:
			raise ValueError("must specify at least one channel in the form --channel-name=IFO=CHANNEL-NAME")
		if options.frame_segments_file is not None and options.data_source != "frames":
			raise ValueError("Can only give --frame-segments-file if --data-source=frames")
		if options.frame_segments_name is not None and options.frame_segments_file is None:
			raise ValueError("Can only specify --frame-segments-name if --frame-segments-file is given")
		if options.data_source == "nds" and (options.nds_host is None or options.nds_port is None):
			raise ValueError("Must specify --nds-host and --nds-port when using --data-source=nds")

		self.channel_dict = channel_dict_from_channel_list(options.channel_name)

		# Set up default dictionary for shared memory partition, and override
		# if the user asks
		self.shm_part_dict = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		if options.shared_memory_partition is not None:
			self.shm_part_dict.update( channel_dict_from_channel_list(options.shared_memory_partition) )

		# Parse the frame segments if they exist
		if options.frame_segments_file is not None:
			self.frame_segments = ligolw_segments.segmenttable_get_by_name(utils.load_filename(options.frame_segments_file, verbose = options.verbose), options.frame_segments_name).coalesce()
		else:
			self.frame_segments = dict([(instrument, None) for instrument in self.channel_dict])

		if options.gps_start_time is not None:
			self.seg = segments.segment(LIGOTimeGPS(options.gps_start_time), LIGOTimeGPS(options.gps_end_time))
			self.seekevent = gst.event_new_seek(1., gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT, gst.SEEK_TYPE_SET, self.seg[0].ns(), gst.SEEK_TYPE_SET, self.seg[1].ns())

		# Set up default dictionary for the DQ (state vector) channel and 
		# override if the user asks
		self.dq_channel_dict = { "H1": "LLD-DQ_VECTOR", "H2": "LLD-DQ_VECTOR","L1": "LLD-DQ_VECTOR", "V1": "LLD-DQ_VECTOR" }
		if options.dq_channel_name is not None:
			self.dq_channel_dict.update( channel_dict_from_channel_list(options.dq_channel_name) )
	
		self.state_vector_on_off_bits = state_vector_on_off_dict_from_bit_lists(options.state_vector_on_bits, options.state_vector_off_bits)
		
		self.frame_cache = options.frame_cache
		self.block_size = options.block_size
		self.data_source = options.data_source
		self.injection_filename = options.injections

		# Store the ndssrc specific options
		if options.data_source == "nds":
			self.nds_host = options.nds_host
			self.nds_port = options.nds_port
			self.nds_channel_type = options.nds_channel_type


def append_options(parser):
	"""
	Append generic data source options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout the project
	for applications that read GW data.
	"""
	group = optparse.OptionGroup(parser, "Data source options", "Use these options to set up the appropriate data source")
	group.add_option("--data-source", metavar = "source", help = "Set the data source from [frames|online|white|silence|AdvVirgo|LIGO|AdvLIGO].  Required")
	group.add_option("--block-size", type="int", metavar = "bytes", default = 16384 * 8 * 512, help = "Data block size to read in bytes. Default 16384 * 8 * 512 (512 seconds of double precision data at 16384 Hz.  This parameter is not used if --data-source=online")
	group.add_option("--frame-cache", metavar = "filename", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).  This is required iff --data-source=frames")
	group.add_option("--gps-start-time", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds. Required unless --data-source=online")
	group.add_option("--gps-end-time", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds.  Required unless --data-source=online")
	group.add_option("--injections", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load injections (optional).")
	group.add_option("--channel-name", metavar = "name", action = "append", help = "Set the name of the channels to process.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--nds-host", metavar = "hostname", help = "Set the remote host or IP address that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-port", metavar = "portnumber", type=int, default=31200, help = "Set the port of the remote host that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-channel-type", metavar = "type", default = "online", help = "Set the port of the remote host that serves nds data. This is required only if --data-source=nds. default==online")	
	group.add_option("--dq-channel-name", metavar = "name", action = "append", help = "Set the name of the data quality (or state vector) channel.  This channel will be used to control the flow of data via the on/off bits.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--shared-memory-partition", metavar = "name", action = "append", help = "Set the name of the shared memory partition for a given instrument.  Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME")
	group.add_option("--frame-segments-file", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load frame segments.  Optional iff --data-source=frames")
	group.add_option("--frame-segments-name", metavar = "name", help = "Set the name of the segments to extract from the segment tables.  Required iff --frame-segments-file is given")
	group.add_option("--state-vector-on-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector on bits to process (optional).  The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online data.")
	group.add_option("--state-vector-off-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector off bits to process (optional).  The default is 0x160 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online data.")
	parser.add_option_group(group)


def mkbasicsrc(pipeline, gw_data_source_info, instrument, verbose = False):
	"""
	All the conditionals and stupid pet tricks for reading real or
	simulated h(t) data in one place.
	"""

	#
	# data source
	#

	# First process fake data or frame data
	if gw_data_source_info.data_source == "white":
		# seek events have to be given to these since the element returned is a tag inject
		src = pipeparts.mkfakesrcseeked(pipeline, instrument, gw_data_source_info.channel_dict[instrument], gw_data_source_info.seekevent, blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "silence":
		# seek events have to be given to these since the element returned is a tag inject
		src = pipeparts.mkfakesrcseeked(pipeline, instrument, gw_data_source_info.channel_dict[instrument], gw_data_source_info.seekevent, blocksize = gw_data_source_info.block_size, wave = 4)
	elif gw_data_source_info.data_source == 'LIGO':
		src = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == 'AdvLIGO':
		src = pipeparts.mkfakeadvLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == 'AdvVirgo':
		src = pipeparts.mkfakeadvvirgosrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "frames":
		if instrument == "V1":
			#FIXME Hack because virgo often just uses "V" in the file names rather than "V1".  We need to sieve on "V"
			src = pipeparts.mkframesrc(pipeline, location = gw_data_source_info.frame_cache, instrument = instrument, cache_src_regex = "V", channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, segment_list = gw_data_source_info.frame_segments[instrument])
		else:
			src = pipeparts.mkframesrc(pipeline, location = gw_data_source_info.frame_cache, instrument = instrument, cache_dsc_regex = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, segment_list = gw_data_source_info.frame_segments[instrument])
	# Next process online data, fake data must be None for this to have gotten this far
	elif gw_data_source_info.data_source == "online":
		# See https://wiki.ligo.org/DAC/ER2DataDistributionPlan#LIGO_Online_DQ_Channel_Specifica
		state_vector_on_bits, state_vector_off_bits = gw_data_source_info.state_vector_on_off_bits[instrument]

		# FIXME make wait_time adjustable through web interface or command line or both
		src = pipeparts.mklvshmsrc(pipeline, shm_name = gw_data_source_info.shm_part_dict[instrument], wait_time = 120)
		src = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = True, skip_bad_files = True)

		# strain
		strain = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND * 60 * 10) # 10 minutes of buffering
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), strain.get_pad("sink"))
		strain = pipeparts.mkaudiorate(pipeline, strain, skip_to_first = True, silent = False)
		@bottle.route("/%s/strain_add_drop.txt" % instrument)
		def strain_add(elem = strain):
			import time
			from pylal.date import XLALUTCToGPS
			t = float(XLALUTCToGPS(time.gmtime()))
			add = elem.get_property("add")
			drop = elem.get_property("drop")
			# FIXME don't hard code the sample rate
			return "%.9f %d %d" % (t, add / 16384., drop / 16384.)

		# state vector
		# FIXME use pipeparts
		statevector = gst.element_factory_make("queue")
		statevector.set_property("max_size_buffers", 0)
		statevector.set_property("max_size_bytes", 0)
		statevector.set_property("max_size_time", gst.SECOND * 60 * 10) # 10 minutes of buffering
		pipeline.add(statevector)
		# FIXME:  don't hard-code channel name
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.dq_channel_dict[instrument]), statevector.get_pad("sink"))
		# FIXME we don't add a signal handler to the statevector audiorate, I assume it should report the same missing samples?
		statevector = pipeparts.mkaudiorate(pipeline, statevector, skip_to_first = True)
		statevector = pipeparts.mkstatevector(pipeline, statevector, required_on = state_vector_on_bits, required_off = state_vector_off_bits)
		@bottle.route("/%s/state_vector_on_off_gap.txt" % instrument)
		def state_vector_state(elem = statevector):
			import time
			from pylal.date import XLALUTCToGPS
			t = float(XLALUTCToGPS(time.gmtime()))
			on = elem.get_property("on-samples")
			off = elem.get_property("off-samples")
			gap = elem.get_property("gap-samples")
			return "%.9f %d %d %d" % (t, on, off, gap)

		# use state vector to gate strain
		src = pipeparts.mkgate(pipeline, strain, threshold = 1, control = statevector, name = "%s_state_vector_gate" % instrument)
		# export state vector state
		if gw_data_source_info.gate_start_callback is not None:
			src.set_property("emit-signals", True)
			src.connect("start", gw_data_source_info.gate_start_callback)
		if gw_data_source_info.gate_stop_callback is not None:
			src.set_property("emit-signals", True)
			src.connect("stop", gw_data_source_info.gate_stop_callback)
	elif gw_data_source_info.data_source == "nds":
		src = pipeparts.mkndssrc(pipeline, gw_data_source_info.nds_host, instrument, gw_data_source_info.channel_dict[instrument], gw_data_source_info.nds_channel_type, blocksize = gw_data_source_info.block_size, port = gw_data_source_info.nds_port)
	else:
		raise ValueError("invalid data_source: %s" % gw_data_source_info.data_source)

	# seek some non-live sources FIXME someday this should go away and seeks
	# should only be done on the pipeline that is why this is separated
	# here
	if gw_data_source_info.data_source in ("LIGO", "AdvLIGO", "AdvVirgo", "frames"):
		#
		# seek the data source if not live
		#

		if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
			raise RuntimeError("Element %s did not want to enter ready state" % src.get_name())
		if not src.send_event(gw_data_source_info.seekevent):
			raise RuntimeError("Element %s did not handle seek event" % src.get_name())

	#
	# provide an audioconvert element to allow Virgo data (which is
	# single-precision) to be adapted into the pipeline
	#

	src = pipeparts.mkaudioconvert(pipeline, src)


	#
	# progress report
	#

	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_src_%s" % instrument)

	#
	# optional injections
	#

	if gw_data_source_info.injection_filename is not None:
		src = pipeparts.mkinjections(pipeline, src, gw_data_source_info.injection_filename)

	#
	# done
	#

	return src


def mkhtgate(pipeline, src, control = None, threshold = 8.0, attack_length = -128, hold_length = -128, name = None):
	"""
	A convenience function to provide thersholds on input data.  This can be used to remove large spikes / glitches etc.
	"""

	# FIXME someday explore a good bandpass filter
	# src = pipeparts.mkaudiochebband(pipeline, src, low_frequency, high_frequency)
	src = pipeparts.mktee(pipeline, src)
	if control is None:
		control = pipeparts.mkqueue(pipeline, src, max_size_time = 0, max_size_bytes = 0, max_size_buffers = 0)
	input = pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND, max_size_bytes = 0, max_size_buffers = 0)
	if name is not None:
		return pipeparts.mkgate(pipeline, input, threshold = threshold, control = control, attack_length = attack_length, hold_length = hold_length, invert_control = True, name = name)
	else:
		return pipeparts.mkgate(pipeline, input, threshold = threshold, control = control, attack_length = attack_length, hold_length = hold_length, invert_control = True)


def mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = None, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, seekevent = None, nxydump_segment = None, track_psd = False, block_duration = None, zero_pad = 0):
	"""Build pipeline stage to whiten and downsample h(t)."""

	#
	# down-sample to highest of target sample rates.  we include a caps
	# filter upstream of the resampler to ensure that this is, infact,
	# *down*-sampling.  if the source time series has a lower sample
	# rate than the highest target sample rate the resampler will
	# become an upsampler, and the result will likely interact poorly
	# with the whitener as it tries to ampify the non-existant
	# high-frequency components, possibly adding significant numerical
	# noise to its output.  if you see errors about being unable to
	# negotiate a format from this stage in the pipeline, it is because
	# you are asking for output sample rates that are higher than the
	# sample rate of your data source.
	#

	quality = 9
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % max(rates))
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove when resampler is patched
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max(rates)))

	#
	# add a reblock element.  to reduce disk I/O gstlal_inspiral asks
	# framesrc to provide enormous buffers, and it helps reduce the RAM
	# pressure of the pipeline by slicing them up.  also, the
	# whitener's gap support isn't 100% yet and giving it smaller input
	# buffers works around the remaining weaknesses (namely that when
	# it sees a gap buffer large enough to drain its internal history,
	# it doesn't know enough to produce a short non-gap buffer to drain
	# its history followed by a gap buffer, it just produces one huge
	# non-gap buffer that's mostly zeros).
	#

	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

	#
	# construct whitener.
	#

	head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
	
	if psd is None:
		# use running average PSD
		head.set_property("psd-mode", 0)
	else:
		# use running psd
		if track_psd:
			head.set_property("psd-mode", 0)
		# use fixed PSD
		else:
			head.set_property("psd-mode", 1)

		#
		# install signal handler to retrieve \Delta f and
		# f_{Nyquist} whenever they are known and/or change,
		# resample the user-supplied PSD, and install it into the
		# whitener.
		#

		def psd_resolution_changed(elem, pspec, psd):
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate and install PSD
			psd = reference_psd.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		head.connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		head.connect_after("notify::delta-f", psd_resolution_changed, psd)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max(rates)))

	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		head = mksegmentsrcgate(pipeline, head, veto_segments, threshold=0.1, seekevent=seekevent, invert_output=True)

	#
	# tee for highest sample rate stream
	#

	head = {max(rates): pipeparts.mktee(pipeline, head)}

	#
	# down-sample whitened time series to remaining target sample rates
	# while applying an amplitude correction to adjust for low-pass
	# filter roll-off.  we also scale by \sqrt{original rate / new
	# rate}.  this is done to preserve the square magnitude of the time
	# series --- the inner product of the time series with itself.
	# really what we want is for
	#
	#	\int v_{1}(t) v_{2}(t) \diff t
	#		\approx \sum v_{1}(t) v_{2}(t) \Delta t
	#
	# to be preserved across different sample rates, i.e. for different
	# \Delta t.  what we do is rescale the time series and ignore
	# \Delta t, so we put 1/2 factor of the ratio of the \Delta t's
	# into the h(t) time series here, and, later, another 1/2 factor
	# into the template when it gets downsampled.
	#
	# by design, the output of the whitener is a unit-variance time
	# series.  however, downsampling it reduces the variance due to the
	# removal of some frequency components.  we require the input to
	# the orthogonal filter banks to be unit variance, therefore a
	# correction factor is applied via an audio amplify element to
	# adjust for the reduction in variance due to the downsampler.
	#

	# FIXME this for loop was reworked to allow the h(t) gate to go after
	# audioresamplers.  There is apparently a cornercase in the
	# audioresample element that is causing a problem
	for rate in sorted(set(rates)):
		if rate < max(rates): # downsample
			head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate)))
			head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head[rate], quality = quality), caps = "audio/x-raw-float, rate=%d" % rate)
			head[rate] = pipeparts.mknofakedisconts(pipeline, head[rate])	# FIXME:  remove when resampler is patched
			head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))

		#
		# optional gate on whitened h(t) amplitude
		#

		if ht_gate_threshold is not None:
			# all h(t) gates are controlled by the same max rate control input.
			head[rate] = mkhtgate(pipeline, head[rate], control = pipeparts.mkqueue(pipeline, head[max(rates)], max_size_time = 0, max_size_bytes = 0, max_size_buffers = 0), threshold = ht_gate_threshold, hold_length = -rate // 4, attack_length = -rate // 4, name = "%s_%d_ht_gate" % (instrument, rate))

			# emit signals so that a user can latch on to them
			head[rate].set_property("emit-signals", True)
	
		head[rate] = pipeparts.mktee(pipeline, head[rate])
	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	return head

