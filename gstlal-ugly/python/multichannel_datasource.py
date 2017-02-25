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

import optparse
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

from gstlal import pipeparts
from gstlal import datasource
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import segments as ligolw_segments
from glue import segments
import lal
from lal import LIGOTimeGPS


## framexmit ports in use on the LDG
# Look-up table to map instrument name to framexmit multicast address and
# port
#
# used in mkbasicsrc() 
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

def channel_dict_from_channel_file(channel_file):
	"""!
	Given a file of channel names with sampling rates, produce a dictionary keyed by ifo:

	The file here comes from the output of a configuration file parser.
	"""

	dict_out = {}
	channel_list = open(channel_file)
	for channel in channel_list:
		ifo, channel_info = channel.split(':')
		channel_name, samp_rate = channel_info.split()		
		dict_out.setdefault(ifo, {})[channel_name] = samp_rate
	channel_list.close()
	return dict_out

def channel_list_from_channel_dict(instrument, channel_dict):
	"""!
	Given a channel dictionary, and instrument tag, will produce a list of channels, in the form
	["H1:channel1", "H1:channel2", ...], given instrument = "H1".
	"""
	out = []
	for channel in channel_dict[instrument]:
		out.append("%s:%s" % (instrument, channel))
	return out	

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
		self.data_sources = set(("framexmit", "lvshm", "white", "silence"))
		self.live_sources = set(("framexmit", "lvshm"))
		assert self.live_sources <= self.data_sources

		# Sanity check the options
		if options.data_source not in self.data_sources:
			raise ValueError("--data-source must be one of %s" % ", ".join(self.data_sources))
		if not options.channel_list:
			raise ValueError("must specify at least one channel in the form --channel-name=IFO=CHANNEL-NAME")

		## A dictionary of the requested channels, e.g., {"H1": {"LDAS-STRAIN": 16384}, "L1": {"LDAS-STRAIN": 16384}}
		self.channel_dict = channel_dict_from_channel_file(options.channel_list)

		## A dictionary for shared memory partition, e.g., {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		self.shm_part_dict = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		if options.shared_memory_partition is not None:
			self.shm_part_dict.update( datasource.channel_dict_from_channel_list(options.shared_memory_partition) )

		## A dictionary of framexmit addresses
		self.framexmit_addr = framexmit_ports["CIT"]
		if options.framexmit_addr is not None:
			self.framexmit_addr.update( datasource.framexmit_dict_from_framexmit_list(options.framexmit_addr) )
		self.framexmit_iface = options.framexmit_iface

		## Analysis segment. Default is None
		self.seg = None

		if options.gps_start_time is not None:
			if options.gps_end_time is None:
				raise ValueError("must provide both --gps-start-time and --gps-end-time")
			if options.data_source in self.live_sources:
				raise ValueError("cannot set --gps-start-time or --gps-end-time with %s" % " or ".join("--data-source=%s" % src for src in sorted(self.live_sources)))
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

		## block size in bytes to read data from disk
		self.block_size = options.block_size
		## Data source, one of python.datasource.DataSourceInfo.data_sources
		self.data_source = options.data_source

def append_options(parser):
	"""!
	Append generic data source options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout the project
	for applications that read GW data.
	
-	--data-source [string]
		Set the data source from [framexmit|lvshm|silence|white].

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

-	--framexmit-addr [string]
		Set the address of the framexmit service.  Can be given
		multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port

-	--framexmit-iface [string]
		Set the address of the framexmit interface.

-	--shared-memory-partition [string]
		Set the name of the shared memory partition for a given instrument.
		Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME

	#### Typical usage case examples

	-# Reading online data via framexmit

		--data-source=framexmit --channel-list=H1=location/to/file

	-# Many other combinations possible, please add some!
	"""
	group = optparse.OptionGroup(parser, "Data source options", "Use these options to set up the appropriate data source")
	group.add_option("--data-source", metavar = "source", help = "Set the data source from [framexmit|lvshm|silence|white].  Required.")
	group.add_option("--block-size", type="int", metavar = "bytes", default = 16384 * 8 * 512, help = "Data block size to read in bytes. Default 16384 * 8 * 512 (512 seconds of double precision data at 16384 Hz.  This parameter is only used if --data-source is one of white, silence.")
	group.add_option("--gps-start-time", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds. Required unless --data-source=lvshm")
	group.add_option("--gps-end-time", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds.  Required unless --data-source=lvshm")
	group.add_option("--channel-list", type='str', metavar = "name", help = "Set the list of the channels to process. File needs to be of the format channel-name[spaces]sampling rate with a new channel on each line. Command given as --channel-list=location/to/file")
	group.add_option("--framexmit-addr", metavar = "name", action = "append", help = "Set the address of the framexmit service.  Can be given multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port")
	group.add_option("--framexmit-iface", metavar = "name", help = "Set the multicast interface address of the framexmit service.")
	group.add_option("--shared-memory-partition", metavar = "name", action = "append", help = "Set the name of the shared memory partition for a given instrument.  Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME")
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
def mkbasicmultisrc(pipeline, data_source_info, instrument, verbose = False):
	"""!
	All the things for reading real or simulated channel data in one place.

	Consult the append_options() function and the DataSourceInfo class

	This src in general supports only one instrument although
	DataSourceInfo contains dictionaries of multi-instrument things.  By
	specifying the instrument when calling this function you will get ony a single
	instrument source.  A code wishing to have multiple basicsrcs will need to call
	this function for each instrument.
	"""

	if data_source_info.data_source == "white":
		src = pipeparts.mkfakesrc(pipeline, instrument, data_source_info.channel_dict[instrument].keys(), blocksize = data_source_info.block_size, volume = 1.0)
	elif data_source_info.data_source == "silence":
		src = pipeparts.mkfakesrc(pipeline, instrument, data_source_info.channel_dict[instrument].keys(), blocksize = data_source_info.block_size, wave = 4)
	elif data_source_info.data_source in ("framexmit", "lvshm"):
		if data_source_info.data_source == "lvshm":
			# FIXME make wait_time adjustable through web interface or command line or both
			src = pipeparts.mklvshmsrc(pipeline, shm_name = data_source_info.shm_part_dict[instrument], wait_time = 120)
		elif data_source_info.data_source == "framexmit":
			src = pipeparts.mkframexmitsrc(pipeline, multicast_iface = data_source_info.framexmit_iface, multicast_group = data_source_info.framexmit_addr[instrument][0], port = data_source_info.framexmit_addr[instrument][1], wait_time = 120)
		else:
			# impossible code path
			raise ValueError(data_source_info.data_source)

		src = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True, channel_list = channel_list_from_channel_dict(instrument, data_source_info.channel_dict))

		# channels
		head = dict.fromkeys(data_source_info.channel_dict[instrument].keys(), None)
		for channel in head:		
			head[channel] = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND* 60 * 1) # 1 minute of buffering
			pipeparts.src_deferred_link(src, "%s:%s" % (instrument, channel), head[channel].get_static_pad("sink"))
		
			# fill in holes, skip duplicate data
			head[channel] = pipeparts.mkaudiorate(pipeline, head[channel], skip_to_first = True, silent = False)

			# 10 minutes of buffering
			head[channel] = pipeparts.mkqueue(pipeline, head[channel], max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * 60 * 10)

	else:
		raise ValueError("invalid data_source: %s" % data_source_info.data_source)

	
	for channel in head:
		#
		# provide an audioconvert element to allow Virgo data (which is
		# single-precision) to be adapted into the pipeline
		#

		head[channel] = pipeparts.mkaudioconvert(pipeline, head[channel])

		#
		# progress report
		#

		if verbose:
			head[channel] = pipeparts.mkprogressreport(pipeline, head[channel], "progress_src_%s_%s" % (instrument, channel))

		#
		# done
		#

	return head

# Unit tests
if __name__ == "__main__":
	import doctest
	doctest.testmod()
