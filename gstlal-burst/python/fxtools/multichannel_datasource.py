#
# Copyright (C) 2017        Sydney J. Chamberlin, Patrick Godwin, Chad Hanna
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
import time
import itertools
import optparse
from ConfigParser import SafeConfigParser

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

from gstlal import pipeparts
from gstlal import datasource
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import segments as ligolw_segments
from ligo import segments
import lal
from lal import LIGOTimeGPS

import numpy

## framexmit ports in use on the LDG
# Look-up table to map instrument name to framexmit multicast address and
# port
#
# used in mkbasicmultisrc() 
# 
# FIXME:  this is only here temporarily while we test this approach to data
# aquisition.  obviously we can't hard-code this stuff
#
framexmit_ports = {
	"CIT": {
		"H1": ("224.3.2.1", 7096),
		"L1": ("224.3.2.2", 7097),
		"V1": ("224.3.2.3", 7098),
	}
}

#
# misc useful functions
#

def channel_dict_from_channel_list(channel_list):
	"""!
	Given a list of channels, produce a dictionary keyed by channel names:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> channel_dict_from_channel_list(["H1:AUX-CHANNEL-NAME_1:2048", "H1:AUX-CHANNEL-NAME-2:512"])
		{'H1:AUX-CHANNEL-NAME_1': {'qhigh': None, 'ifo': 'H1', 'flow': None, 'fsamp': 2048.0, 'fhigh': None, 'frametype': None}, 'H1:AUX-CHANNEL-NAME-2': {'qhigh': None, 'ifo': 'H1', 'flow': None, 'fsamp': 512.0, 'fhigh': None, 'frametype': None}}
	"""

	channel_dict = {}
	for channel in channel_list:
		ifo, channel_info, fsamp = channel.split(':')
		channel_name = ifo + ":" + channel_info
		channel_dict[channel_name] = {'fsamp': float(fsamp),
					      'ifo': ifo,
					      'flow': None,
					      'fhigh': None,
					      'qhigh' : None,
					      'frametype' : None}
	return channel_dict

def channel_dict_from_channel_file(channel_file):
	"""!
	Given a file of channel names with sampling rates, produce a dictionary keyed by ifo:

	The file here comes from the output of a configuration file parser.
	"""

	channel_dict = {}
	channel_list = open(channel_file)
	for channel in channel_list:
		channel_name, fsamp = channel.split()
		ifo, channel_info = channel.split(':')
		channel_dict[channel_name] = {'fsamp': float(fsamp),
					      'ifo': ifo,
					      'flow': None,
					      'fhigh': None,
					      'qhigh' : None,
					      'frametype' : None}

	channel_list.close()
	return channel_dict

def channel_dict_from_channel_ini(options):
	"""!
	Given a channel list INI, produces a dictionary keyed by ifo, filtered by frame type.
	"""

	channel_dict = {}

	# frame types considered
	included_frame_types = set(("H1_R", "L1_R", "H1_lldetchar", "L1_lldetchar"))

	# known/permissible values of safety and fidelity flags
	known_safety   = set(("safe", "unsafe", "unsafeabove2kHz", "unknown"))
	known_fidelity = set(("clean", "flat", "glitchy", "unknown"))
	
	# read in channel list
	config = SafeConfigParser()
	config.read(options.channel_list)
	
	# filter out channels by frame type	
	sections = []
	for name in config.sections():
		if config.get(name, 'frametype') in included_frame_types:
			sections.append(name)

	# specify which channels are considered
	if options.section_include:
		section_include = [section.replace('_', ' ') for section in options.section_include]
	else:
		section_include = sections

	channel_include = options.safe_channel_include + options.unsafe_channel_include

	# generate dictionary of channels
	for name in sections:

		# extract low frequency, high Q
		flow = config.getfloat(name, 'flow')
		qhigh = config.getfloat(name, 'qhigh')

		# figure out whether to use Nyquist for each channel or a specific limit
		fhigh  = config.get(name, 'fhigh')
		use_nyquist = fhigh == "Nyquist"
		if not use_nyquist:
			fhigh = float(fhigh)

		# set up each channel
		for channel in config.get(name, 'channels').strip().split('\n'):

			# parse out expected format for each channel
			channel = channel.split()

			if len(channel)==2: # backward compatibility with old format
				channel, fsamp = channel
				fsamp = int(fsamp)
				safety = "unknown"
				fidelity = "unknown"

			elif len(channel)==4: # expected format
				channel, fsamp, safety, fidelity = channel
				fsamp = int(fsamp)

			else:
				raise SyntaxError( 'could not parse channel : %s'%(''.join(channel)) )

		    #-----------------------------------------

			### check that safety and fidelity are permissible values
			assert safety   in known_safety,   'safety=%s is not understood. Must be one of %s'%(safety, ", ".join(known_safety))
			assert fidelity in known_fidelity, 'fidelity=%s is not understood. Must be one of %s'%(fidelity, ", ".join(known_fidelity))

			# conditions on whether or not we want to include this channel
			if name in section_include or channel in channel_include:
				if (safety in options.safety_include and fidelity not in options.fidelity_exclude) or channel in options.unsafe_channel_include:

					# add ifo, channel name & omicron parameters to dict
					channel_name = channel
					ifo,_  = channel.split(':')
					if use_nyquist:
						fhigh = fsamp/2.

					channel_dict[channel_name] = {'fsamp': fsamp, 'ifo': ifo, 'flow': flow, 'fhigh': fhigh, 'qhigh' : qhigh}

	return channel_dict				



def partition_channels_to_equal_subsets(channel_dict, max_streams, min_sample_rate, max_sample_rate):
	"""!
	Given a channel dictionary, will produce partitions of channel subsets where the number of channels
	in each partition is equal (except possibly the last partition). This is given max_streams,
	and well as max and min sample rates enforced to determine the number of streams that a particular
	channel will generate.

	Returns a list of disjoint channel lists.
	"""
	# determine how many streams a single channel will produce when split up into multiple frequency bands
	# and separate them based on this criterion
	channel_streams = {}

	for channel in channel_dict.keys():
		sample_rate = int(channel_dict[channel]['fsamp'])
		max_rate = min(max_sample_rate, sample_rate)
		min_rate = min(min_sample_rate, max_rate)
		n_rates = int(numpy.log2(max_rate/min_rate) + 1)

		channel_streams.setdefault(n_rates, []).append((n_rates, channel))

	# find relative probabilities in each bin
	total = sum((len(channel_streams[n]) for n in channel_streams.keys()))
	p_relative = {n: (len(channel_streams[n]) / float(total)) for n in channel_streams.keys()}

	# figure out total number of channels needed per subset
	num_streams = {n: int(numpy.ceil(p_relative[n] * max_streams)) for n in channel_streams.keys()}
	num_channels = {n: int(numpy.ceil(num_streams[n] / float(n))) for n in num_streams.keys()}

	# force less sampling from the lowest bins (16Hz and 32Hz) at the beginning
	# to reduce the number of channels in each subset
	rates = sorted(channel_streams.keys())
	if rates[0] == 1:
		num_channels[1] = 1
	max_channels = sum((num_channels[n] for n in num_channels.keys()))

	# generate a round-robin type way to sample from
	rates2sample = itertools.cycle(n for n in channel_streams.keys() for i in range(int(numpy.round(p_relative[n] * max_channels))))

	# generate channel subsets
	subsets = []
	total = sum((len(channel_streams[n]) for n in channel_streams.keys()))
	while total > 0:
		subset = []
		while len(subset) < max_channels and total > 0:
			rate = next(rates2sample)
			while not channel_streams[rate]:
				# recalculate probabilities and rates2sample
				p_relative = {n: (len(channel_streams[n]) / float(total)) for n in channel_streams.keys()}
				rates2sample = itertools.cycle(n for n in channel_streams.keys() for i in range(int(numpy.round(p_relative[n] * max_channels))))
				rate = next(rates2sample)

			subset.append(channel_streams[rate].pop()[1])
			total -= 1

		subsets.append(subset)

	return subsets

def partition_channels_to_subsets(channel_dict, max_streams, min_sample_rate, max_sample_rate):
	"""!
	Given a channel dictionary, will produce roughly equal partitions of channel subsets, given max_streams,
	and well as max and min sample rates enforced to determine the number of streams that a particular
	channel will generate.

	Returns a list of disjoint channel lists.
	"""
	# determine how many streams a single channel will produce when split up into multiple frequency bands
	channel_streams = []

	for channel in channel_dict.keys():
		sample_rate = int(channel_dict[channel]['fsamp'])
		max_rate = min(max_sample_rate, sample_rate)
		min_rate = min(min_sample_rate, max_rate)
		n_rates = int(numpy.log2(max_rate/min_rate) + 1)

		channel_streams.append((n_rates, channel))

	return [subset for subset in partition_list(channel_streams, max_streams)]

def partition_list(lst, target_sum):
	"""!
	Partition list to roughly equal partitioned chunks based on a target sum,
	given a list with items in the form (int, value), where ints are used to determine partitions.

	Returns a sublist with items value.
	"""
	total_sum = sum(item[0] for item in lst)
	chunks = numpy.ceil(total_sum/float(target_sum))
	avg_sum = total_sum/float(chunks)

	chunks_yielded = 0
	chunk = []
	chunksum = 0
	sum_of_seen = 0

	for i, item in enumerate(lst):

		# if only one chunk left to process, yield rest of list
		if chunks - chunks_yielded == 1:
			yield chunk + [x[1] for x in lst[i:]]
			raise StopIteration

		to_yield = chunks - chunks_yielded
		chunks_left = len(lst) - i

		# yield remaining list in single item chunks
		if to_yield > chunks_left:
			if chunk:
				yield chunk
			for x in lst[i:]:
				yield [x[1]]
			raise StopIteration

		sum_of_seen += item[0]

		# if target sum is less than the average, add another item to chunk
		if chunksum < avg_sum:
			chunk.append(item[1])
			chunksum += item[0]

		# else, yield the chunk, and update expected sum since this chunk isn't perfectly partitioned
		else:
			yield chunk
			avg_sum = (total_sum - sum_of_seen)/(to_yield - 1)
			chunks_yielded += 1
			chunksum = item[0]
			chunk = [item[1]]

class DataSourceInfo(object):
	"""!
	Hold the data associated with data source command lines.
	"""
	## See datasource.append_options()
	def __init__(self, options):
		"""!
		Initialize a DataSourceInfo class instance from command line options specified by append_options()
		""" 

		## A list of possible, valid data sources ("frames", "framexmit", "lvshm", "white", "silence")
		self.data_sources = set(("framexmit", "lvshm", "frames", "white", "silence", "white_live"))
		self.live_sources = set(("framexmit", "lvshm", "white_live"))
		assert self.live_sources <= self.data_sources

		# Sanity check the options
		if options.data_source not in self.data_sources:
			raise ValueError("--data-source must be one of %s" % ", ".join(self.data_sources))
		if options.data_source == "frames" and options.frame_cache is None:
			raise ValueError("--frame-cache must be specified when using --data-source=frames")
		if options.frame_segments_file is not None and options.data_source != "frames":
			raise ValueError("can only give --frame-segments-file if --data-source=frames")
		if options.frame_segments_name is not None and options.frame_segments_file is None:
			raise ValueError("can only specify --frame-segments-name if --frame-segments-file is given")	
		if not (options.channel_list or options.channel_name):
			raise ValueError("must specify a channel list in the form --channel-list=/path/to/file or --channel-name=H1:AUX-CHANNEL-NAME:RATE --channel-name=H1:SOMETHING-ELSE:RATE")
		if (options.channel_list and options.channel_name):
			raise ValueError("must specify a channel list in the form --channel-list=/path/to/file or --channel-name=H1:AUX-CHANNEL-NAME:RATE --channel-name=H1:SOMETHING-ELSE:RATE")

		## Generate a dictionary of requested channels from channel INI file
		
		# known/permissible values of safety and fidelity flags
		self.known_safety   = set(("safe", "unsafe", "unsafeabove2kHz", "unknown"))
		self.known_fidelity = set(("clean", "flat", "glitchy", "unknown"))

		# ensure safety and fidelity options are valid
		options.safety_include = set(options.safety_include)
		options.fidelity_exclude = set(options.fidelity_exclude)

		for safety in options.safety_include:
			assert safety in self.known_safety, '--safety-include=%s is not understood. Must be one of %s'%(safety, ", ".join(self.known_safety))

		for fidelity in options.fidelity_exclude:
			assert fidelity in self.known_fidelity, '--fidelity-exclude=%s is not understood. Must be one of %s'%(fidelity, ", ".join(self.known_fidelity))

		# dictionary of the requested channels, e.g., {"H1:LDAS-STRAIN": 16384, "H1:ODC-LARM": 2048}
		if options.channel_list:
			name, self.extension = options.channel_list.rsplit('.', 1)
			if self.extension == 'ini':
				self.channel_dict = channel_dict_from_channel_ini(options)
			else:
				self.channel_dict = channel_dict_from_channel_file(options.channel_list)
		elif options.channel_name:
			self.extension = 'none'
			self.channel_dict = channel_dict_from_channel_list(options.channel_name)

		# set instrument; it is assumed all channels from a given channel list are from the same instrument
		self.instrument = self.channel_dict[next(iter(self.channel_dict))]['ifo']

		# set the maximum number of streams to be run by a single pipeline.
		self.max_streams = options.max_streams

		# set the frequency ranges considered by channels with splitting into multiple frequency bands.
		# If channel sampling rate doesn't fall within this range, it will not be split into multiple bands.
		self.max_sample_rate = options.max_sample_rate
		self.min_sample_rate = options.min_sample_rate

		# split up channels requested into partitions for serial processing
		if options.equal_subsets:
			self.channel_subsets = partition_channels_to_equal_subsets(self.channel_dict, self.max_streams, self.min_sample_rate, self.max_sample_rate)
		else:
			self.channel_subsets = partition_channels_to_subsets(self.channel_dict, self.max_streams, self.min_sample_rate, self.max_sample_rate)

		## A dictionary for shared memory partition, e.g., {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		self.shm_part_dict = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		if options.shared_memory_partition is not None:
			self.shm_part_dict.update( datasource.channel_dict_from_channel_list(options.shared_memory_partition) )

		## options for shared memory
		self.shm_assumed_duration = options.shared_memory_assumed_duration
		self.shm_block_size = options.shared_memory_block_size # NOTE: should this be incorporated into options.block_size? currently only used for offline data sources

		## A dictionary of framexmit addresses
		self.framexmit_addr = framexmit_ports["CIT"]
		if options.framexmit_addr is not None:
			self.framexmit_addr.update( datasource.framexmit_dict_from_framexmit_list(options.framexmit_addr) )
		self.framexmit_iface = options.framexmit_iface

		## Analysis segment. Default is None
		self.seg = None
		
		## Set latency output
		self.latency_output = options.latency_output

		if options.gps_start_time is not None:
			if options.gps_end_time is None:
				raise ValueError("must provide both --gps-start-time and --gps-end-time")
			try:
				start = LIGOTimeGPS(options.gps_start_time)
			except ValueError:
				raise ValueError("invalid --gps-start-time '%s'" % options.gps_start_time)
			try:
				end = LIGOTimeGPS(options.gps_end_time)
			except ValueError:
				raise ValueError("invalid --gps-end-time '%s'" % options.gps_end_time)
			if start >= end:
				raise ValueError("--gps-start-time must be < --gps-end-time: %s < %s" % (options.gps_start_time, options.gps_end_time))
			## Segment from gps start and stop time if given
			self.seg = segments.segment(LIGOTimeGPS(options.gps_start_time), LIGOTimeGPS(options.gps_end_time))
		elif options.gps_end_time is not None:
			raise ValueError("must provide both --gps-start-time and --gps-end-time")
		elif options.data_source not in self.live_sources:
			raise ValueError("--gps-start-time and --gps-end-time must be specified when --data-source not one of %s" % ", ".join(sorted(self.live_sources)))
		
		if options.frame_segments_file is not None:
			## Frame segments from a user defined file
			self.frame_segments = ligolw_segments.segmenttable_get_by_name(ligolw_utils.load_filename(options.frame_segments_file, contenthandler=ligolw_segments.LIGOLWContentHandler), options.frame_segments_name).coalesce()
			if self.seg is not None:
				# Clip frame segments to seek segment if it
				# exists (not required, just saves some
				# memory and I/O overhead)
				self.frame_segments = segments.segmentlistdict((instrument, seglist & segments.segmentlist([self.seg])) for instrument, seglist in self.frame_segments.items())
		else:
			## if no frame segments provided, set them to an empty segment list dictionary
			self.frame_segments = segments.segmentlistdict({self.instrument: None})
		
		## frame cache file
		self.frame_cache = options.frame_cache
		## block size in bytes to read data from disk
		self.block_size = options.block_size
		## Data source, one of python.datasource.DataSourceInfo.data_sources
		self.data_source = options.data_source

		# FIXME: this is ugly, but we have to protect against busted shared memory partitions
		if self.data_source == "lvshm":
			import subprocess
			subprocess.call(["smrepair", "--bufmode", "5", self.shm_part_dict[self.instrument]])

def append_options(parser):
	"""!
	Append generic data source options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout the project
	for applications that read GW data.
	
-	--data-source [string]
		Set the data source from [framexmit|lvshm|frames|silence|white|white_live].

-	--block-size [int] (bytes)
		Data block size to read in bytes. Default 16384 * 8 * 512 which is 512 seconds of double
		precision data at 16384 Hz. This parameter is only used if --data-source is one of
		white, silence, AdvVirgo, LIGO, AdvLIGO, nds.

-	--gps-start-time [int] (seconds)
		Set the start time of the segment to analyze in GPS seconds.
		Required unless --data-source in lvshm,framexmit

-	--gps-end-time  [int] (seconds)
		Set the end time of the segment to analyze in GPS seconds.  
		Required unless --data-source in lvshm,framexmit

-	--channel-list [string]
		Set the list of the channels to process.
		File needs to be in format channel-name[spaces]sampling_rate with a new channel in each line.
		Command given as --channel-list=location/to/file.

-	--channel-name [string]
		Set the name of the channels to process.
		Can be given multiple times as --channel-name=IFO:AUX-CHANNEL-NAME:RATE

-	--max-streams [int]
		Set the maximum number of streams to process at a given time (num_channels * num_rates = num_streams).
		Used to split up channels given into roughly equal subsets to be processed in sequence.

-	--equal-subsets
		If set, forces an equal number of channels processed per channel subset.

-	--max-sampling-rate [int]
		Maximum sampling rate for a given channel.
		If a given channel has a higher native sampling rate, it will be downsampled to this target rate.

-	--min-sampling-rate [int]
		Minimum sampling rate for a given channel when splitting a given channel into multiple frequency bands.
		If a channel has a lower sampling rate than this minimum, however, it will not be upsampled to this sampling rate.

-	--framexmit-addr [string]
		Set the address of the framexmit service.  Can be given
		multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port

-	--framexmit-iface [string]
		Set the address of the framexmit interface.

-	--shared-memory-partition [string]
		Set the name of the shared memory partition for a given instrument.
		Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME

-	--shared-memory-assumed-duration [int]
		Set the assumed span of files in seconds. Default = 4 seconds.

-	--shared-memory-block-size [int]
		Set the byte size to read per buffer. Default = 4096 bytes.

-	--frame-type [string]
		Set the frame type required by the channels being used.

-	--frame-segments-file [filename]
		Set the name of the LIGO light-weight XML file from which to load frame segments.
		Optional iff --data-source is frames

-	--frame-segments-name [string]
		Set the name of the segments to extract from the segment tables.
		Required iff --frame-segments-file is given

-	--section-include [string]
		Set the channel sections to be included from the INI file. Can be given multiple times. Pass in spaces as underscores instead. If not specified, assumed to include all sections.

-	--safety-include [string]
		Set the safety values for channels to be included from the INI file. Can be given multiple times. Default = "safe".

-	--fidelity-exclude [string]
		Set the fidelity values to be excluded from the INI file. Can be given multiple times.
	
-	--safe-channel-include [string]
		Set the channel names to be included from the INI file. Can be given multiple times. If not specified, assumed to include all channels.

-	--unsafe-channel-include [string]
		Include this channel when reading the INI file, disregarding safety information (requires exact match). Can be repeated.

-	--latency-output
		Set whether to print out latency (in seconds) at various stages of the pipeline.

	#### Typical usage case examples

	-# Reading data from frames

		--data-source=frames --gps-start-time=999999000 --gps-end-time=999999999 --channel-name=H1:AUX-CHANNEL-NAME:RATE

	-# Reading online data via framexmit

		--data-source=framexmit --channel-list=H1=location/to/file

	-# Many other combinations possible, please add some!
	"""
	group = optparse.OptionGroup(parser, "Data Source Options", "Use these options to set up the appropriate data source")
	group.add_option("--data-source", metavar = "source", help = "Set the data source from [framexmit|lvshm|frames|silence|white|white_live].  Required.")
	group.add_option("--block-size", type="int", metavar = "bytes", default = 16384 * 8 * 512, help = "Data block size to read in bytes. Default 16384 * 8 * 512 (512 seconds of double precision data at 16384 Hz.  This parameter is only used if --data-source is one of white, silence.")
	group.add_option("--gps-start-time", type="int", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds. Required unless --data-source=lvshm")
	group.add_option("--gps-end-time", type="int", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds.  Required unless --data-source=lvshm")
	group.add_option("--frame-cache", metavar = "filename", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).  This is required iff --data-source=frames")
	group.add_option("--max-streams", type = "int", default = 50, help = "Maximum number of streams to process for a given pipeline at once. Used to split up channel lists into subsets that can then be processed in serial. Default = 50.")
	group.add_option("--equal-subsets", action = "store_true", help = "If set, forces an equal number of channels processed per channel subset.")
	group.add_option("--max-sample-rate", type = "int", default = 4096, help = "Maximum sampling rate for a given channel. If a given channel has a higher native sampling rate, it will be downsampled to this target rate. Default = 4096.")
	group.add_option("--min-sample-rate", type = "int", default = 32, help = "Minimum sampling rate for a given channel when splitting a given channel into multiple frequency bands. If a channel has a lower sampling rate than this minimum, however, it will not be upsampled to this sampling rate. Default = 32.")
	group.add_option("--framexmit-addr", metavar = "name", action = "append", help = "Set the address of the framexmit service.  Can be given multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port")
	group.add_option("--framexmit-iface", metavar = "name", help = "Set the multicast interface address of the framexmit service.")
	group.add_option("--shared-memory-partition", metavar = "name", action = "append", help = "Set the name of the shared memory partition for a given instrument.  Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME")
	group.add_option("--shared-memory-assumed-duration", type = "int", default = 4, help = "Set the assumed span of files in seconds. Default = 4.")
	group.add_option("--shared-memory-block-size", type = "int", default = 4096, help = "Set the byte size to read per buffer. Default = 4096.")
	group.add_option("--frame-type", type="string", metavar = "name", help = "Include only those channels with the frame type given.")
	group.add_option("--frame-segments-file", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load frame segments.  Optional iff --data-source=frames")
	group.add_option("--frame-segments-name", metavar = "name", help = "Set the name of the segments to extract from the segment tables.  Required iff --frame-segments-file is given")	
	group.add_option("--latency-output", action = "store_true", help = "Print out latency output (s) at different stages of the pipeline (measured as current time - buffer time).")
	parser.add_option_group(group)

	group = optparse.OptionGroup(parser, "Channel Options", "Settings used for deciding which auxiliary channels to process.")
	group.add_option("--channel-list", type="string", metavar = "name", help = "Set the list of the channels to process. Command given as --channel-list=location/to/file")
	group.add_option("--channel-name", metavar = "name", action = "append", help = "Set the name of the channels to process.  Can be given multiple times as --channel-name=IFO:AUX-CHANNEL-NAME:RATE")
	group.add_option("--section-include", default=[], type="string", action="append", help="Set the channel sections to be included from the INI file. Can be given multiple times. Pass in spaces as underscores instead. If not specified, assumed to include all sections")
	group.add_option("--safety-include", default=["safe"], type="string", action="append", help="Set the safety values for channels to be included from the INI file. Can be given multiple times. Default = 'safe'.")
	group.add_option("--fidelity-exclude", default=[], type="string", action="append", help="Set the fidelity values for channels to be excluded from the INI file. Can supply multiple values by repeating this argument. Each must be on of (add here)")
	group.add_option("--safe-channel-include", default=[], action="append", type="string", help="Include this channel when reading the INI file (requires exact match). Can be repeated. If not specified, assume to include all channels.")
	group.add_option("--unsafe-channel-include", default=[], action="append", type="string", help="Include this channel when reading the INI file, disregarding safety information (requires exact match). Can be repeated.")
	parser.add_option_group(group)

##
# _Gstreamer graph describing this function:_
#
# @dot
# digraph mkbasicsrc {
#      compound=true;
#      node [shape=record fontsize=10 fontname="Verdana"];
#      subgraph clusterfakesrc {
#              fake_0 [label="fakesrc: white, silence", URL="\ref pipeparts.mkfakesrc()"];
#              color=black;
#              label="Possible path #1";
#      }
#	subgraph clusteronline {
#		color=black;
#		online_0 [label="lvshmsrc|framexmit", URL="\ref pipeparts.mklvshmsrc()"];
#		online_1 [label ="framecppchanneldemux", URL="\ref pipeparts.mkframecppchanneldemux()"];
#		online_2a [label ="channel 1 queue", URL="\ref pipeparts.mkqueue()"];
#		online_2b [label ="channel 2 queue", URL="\ref pipeparts.mkqueue()"];
#		online_2c [label ="channel 3 queue", URL="\ref pipeparts.mkqueue()"];
#		online_3a [label ="audiorate 1", URL="\ref pipeparts.mkaudiorate()"];
#		online_3b [label ="audiorate 2", URL="\ref pipeparts.mkaudiorate()"];
#		online_3c [label ="audiorate 3", URL="\ref pipeparts.mkaudiorate()"];
#		online_0 -> online_1;
#		online_1 -> online_2a;
#		online_1 -> online_2b;
#		online_1 -> online_2c;
#		online_2a -> online_3a;
#		online_2b -> online_3b;
#		online_2c -> online_3c;
#		label="Possible path #2";
#	}
#	audioconv [label="audioconvert", URL="\ref pipeparts.mkaudioconvert()"];
#	progress [label="progressreport (if verbose)", style=filled, color=lightgrey, URL="\ref pipeparts.mkprogressreport()"];
#	sim [label="lalsimulation (if injections requested)", style=filled, color=lightgrey, URL="\ref pipeparts.mkinjections()"];
#	queue [label="queue (if injections requested)", style=filled, color=lightgrey, URL="\ref pipeparts.mkqueue()"];
#
#	// The connections
#	fake_0 -> audioconv [ltail=clusterfakesrc];
#	frames_4 -> audioconv [ltail=clusterframes];
#	online_6 -> audioconv [ltail=clusteronline];
#	nds_0 -> audioconv [ltail=clusternds];
#	audioconv -> progress -> sim -> queue -> "?"
# }
# @enddot
#
#
def mkbasicmultisrc(pipeline, data_source_info, channels, verbose = False):
	"""!
	All the things for reading real or simulated channel data in one place.

	Consult the append_options() function and the DataSourceInfo class

	This src in general supports only one instrument although
	DataSourceInfo contains dictionaries of multi-instrument things.  By
	specifying the channels when calling this function you will only process
	the channels specified. A code wishing to have multiple basicmultisrcs
	will need to call this multiple times with different sets of channels specified.
	"""

	if data_source_info.data_source == "white":
		head = {channel : pipeparts.mkfakesrc(pipeline, instrument = data_source_info.instrument, channel_name = channel, volume = 1.0, rate = data_source_info.channel_dict[channel]['fsamp'], timestamp_offset = int(data_source_info.seg[0]) * Gst.SECOND) for channel in channels}
	elif data_source_info.data_source == "silence":
		head = {channel : pipeparts.mkfakesrc(pipeline, instrument = data_source_info.instrument, channel_name = channel, rate = data_source_info.channel_dict[channel]['fsamp'], timestamp_offset = int(data_source_info.seg[0]) * Gst.SECOND) for channel in channels}
	elif data_source_info.data_source == "white_live":
		head = {channel : pipeparts.mkfakesrc(pipeline, instrument = data_source_info.instrument, channel_name = channel, volume = 1.0, is_live = True, rate = data_source_info.channel_dict[channel]['fsamp'], timestamp_offset = int(data_source_info.seg[0]) * Gst.SECOND) for channel in channels}
	elif data_source_info.data_source == "frames":
		src = pipeparts.mklalcachesrc(pipeline, location = data_source_info.frame_cache, cache_src_regex = data_source_info.instrument[0], cache_dsc_regex = data_source_info.instrument)
		demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True, channel_list = channels)
		# allow frame reading and decoding to occur in a different
		# thread
		head = dict.fromkeys(channels, None)
		for channel in head:	
			head[channel] = pipeparts.mkreblock(pipeline, None, block_duration = 4 * Gst.SECOND)
			pipeparts.src_deferred_link(demux, channel, head[channel].get_static_pad("sink"))
			head[channel] = pipeparts.mkqueue(pipeline, head[channel], max_size_buffers = 0, max_size_bytes = 0, max_size_time = 8 * Gst.SECOND)
			if data_source_info.frame_segments[data_source_info.instrument]:
				# FIXME:  make segmentsrc generate segment samples at the channel sample rate?
				# FIXME:  make gate leaky when I'm certain that will work.
				head[channel] = pipeparts.mkgate(pipeline, head[channel], threshold = 1, control = pipeparts.mksegmentsrc(pipeline, data_source_info.frame_segments[data_source_info.instrument]), name = "%s_frame_segments_gate" % channel)
				pipeparts.framecpp_channeldemux_check_segments.set_probe(head[channel].get_static_pad("src"), data_source_info.frame_segments[data_source_info.instrument])
		
			# fill in holes, skip duplicate data
			head[channel] = pipeparts.mkaudiorate(pipeline, head[channel], skip_to_first = True, silent = False)

	elif data_source_info.data_source in ("framexmit", "lvshm", "white_live"):
		if data_source_info.data_source == "lvshm":
			# FIXME make wait_time adjustable through web interface or command line or both
			src = pipeparts.mklvshmsrc(pipeline, shm_name = data_source_info.shm_part_dict[data_source_info.instrument], assumed_duration = data_source_info.shm_assumed_duration, blocksize = data_source_info.shm_block_size, wait_time = 120)
		elif data_source_info.data_source == "framexmit":
			src = pipeparts.mkframexmitsrc(pipeline, multicast_iface = data_source_info.framexmit_iface, multicast_group = data_source_info.framexmit_addr[data_source_info.instrument][0], port = data_source_info.framexmit_addr[data_source_info.instrument][1], wait_time = 120)
		else:
			# impossible code path
			raise ValueError(data_source_info.data_source)

		demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True, channel_list = channels)

		# channels
		head = dict.fromkeys(channels, None)
		for channel in head:		
			head[channel] = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND* 60 * 1) # 1 minute of buffering
			pipeparts.src_deferred_link(demux, channel, head[channel].get_static_pad("sink"))
			if data_source_info.latency_output:
				head[channel] = pipeparts.mklatency(pipeline, head[channel], name = 'stage1_afterFrameXmit_%s' % channel)

			# fill in holes, skip duplicate data
			head[channel] = pipeparts.mkaudiorate(pipeline, head[channel], skip_to_first = True, silent = False)

			# 10 minutes of buffering
			head[channel] = pipeparts.mkqueue(pipeline, head[channel], max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * 60 * 10)

	else:
		raise ValueError("invalid data_source: %s" % data_source_info.data_source)

	for channel in head:
		head[channel] = pipeparts.mkaudioconvert(pipeline, head[channel])
		# progress report
		if verbose:
			head[channel] = pipeparts.mkprogressreport(pipeline, head[channel], "datasource_progress_for_%s" % channel)
		 
	return head

# Unit tests
if __name__ == "__main__":
	import doctest
	doctest.testmod()
