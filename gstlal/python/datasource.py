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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

## 
# @file
#
# A file that contains the datasource module code
#

##
# @package python.datasource
#
# datasource module

import sys
import optparse

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
from glue.ligolw import lsctables
from glue import segments
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS


## #### ContentHandler
# A stub to wrap ligolw.LIGOLWContentHandler for now
class ContentHandler(ligolw.LIGOLWContentHandler):
	pass
lsctables.use_in(ContentHandler)


#
# Misc useful functions
#

## #### A dictionary of channel names from a list
#  _Python doc string_:
def channel_dict_from_channel_list(channel_list):
	"""!
	Given a list of channels like this
	
		channel_list = ["H1=LSC-STRAIN", H2="SOMETHING-ELSE"]

	produce a dictionary keyed by ifo of channel names:

		{"H1":"LSC-STRAIN", "H2":"SOMETHING-ELSE"}
	"""
	return dict(instrument_channel.split("=") for instrument_channel in channel_list)


## #### A string of channel names from a dictionary
#  _Python doc string_:
def pipeline_channel_list_from_channel_dict(channel_dict, ifos = None, opt = "channel-name"):
	"""!
	Creates a string of channel names options from a dictionary keyed by ifos.
	Output looks like

		--channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN

	- override --channel-name with a different option by setting opt.
	- restrict the ifo keys to a subset of the channel_dict by 
	  setting ifos
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


## #### Default dictionary of state vector on/off bits by ifo
# Used as the default argument to state_vector_on_off_dict_from_bit_lists()
state_vector_on_off_dict = {
	"H1" : [0x7, 0x160], 
	"H2" : [0x7, 0x160],
	"L1" : [0x7, 0x160],
	"V1" : [0x67, 0x100]
}


## #### A dictionary of state vector bits from a list
#  _Python doc string_:
def state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list, state_vector_on_off_dict = state_vector_on_off_dict):
	"""!
	Produce a dictionary (keyed by detector) of on / off bit tuples from a
	list provided on the command line. e.g., given:

		on_bit_list = ["V1=7", "H1=7", "L1=7"]
		off_bit_list = ["V1=256", "H1=352", "L1=352"]

	produce:

		state_vector_on_off_dict = {"H1":[0x7, 0x160], "H2":[0x7, 0x160], "L1":[0x7, 0x160], "V1":[0x67, 0x100]}
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


## #### A list of state vector command line arguments from a dictionary
#  _Python doc string_:
def state_vector_on_off_list_from_bits_dict(bit_dict):
	"""!
	Produce a commandline useful list from a dictionary of on / off state
	vector bits keyed by detector. e.g., given

		state_vector_on_off_dict = {"H1":[0x7, 0x160], "H2":[0x7, 0x160], "L1":[0x7, 0x160], "V1":[0x67, 0x100]}

	produce:

		--state-vector-off-bits V1=256 --state-vector-off-bits=H1=352 --state-vector-off-bits=L1=352  --state-vector-on-bits V1=7 --state-vector-on-bits=H1=7 --state-vector-on-bits=L1=7
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


## #### framexmit ports in use on the LDG
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

## #### Main organizational data class of this module
#  _Python doc string_:
class GWDataSourceInfo(object):
	"""!
	Hold the data associated with data source command lines.
	"""
	## See python.datasource.append_options()
	def __init__(self, options):
		"""!
		Initialize a GWDataSourceInfo class instance from command line options specified by append_options()
		""" 

		## A list of possible, valid data sources ("frames", "framexmit", "lvshm", "nds", "white", "silence", "AdvVirgo", "LIGO", "AdvLIGO")
		self.data_sources = ("frames", "framexmit", "lvshm", "nds", "white", "silence", "AdvVirgo", "LIGO", "AdvLIGO")

		## Callback function to handle the "start" signals from the gate
		self.gate_start_callback = None

		## Callback function to handle the "stop" signals from the gate
		self.gate_stop_callback = None

		# Sanity check the options
		if options.data_source not in self.data_sources:
			raise ValueError("--data-source must be one of %s" % ", ".join(self.data_sources))
		if options.data_source == "frames" and options.frame_cache is None:
			raise ValueError("--frame-cache must be specified when using --data-source=frames")
		if (options.gps_start_time is None or options.gps_end_time is None) and options.data_source == "frames":
			raise ValueError("--gps-start-time and --gps-end-time must be specified unless --data-source=lvshm")
		if len(options.channel_name) == 0:
			raise ValueError("must specify at least one channel in the form --channel-name=IFO=CHANNEL-NAME")
		if options.frame_segments_file is not None and options.data_source != "frames":
			raise ValueError("Can only give --frame-segments-file if --data-source=frames")
		if options.frame_segments_name is not None and options.frame_segments_file is None:
			raise ValueError("Can only specify --frame-segments-name if --frame-segments-file is given")
		if options.data_source == "nds" and (options.nds_host is None or options.nds_port is None):
			raise ValueError("Must specify --nds-host and --nds-port when using --data-source=nds")

		## A dictionary of the requested channels, e.g., {"H1":"LDAS-STRAIN", "L1":"LDAS-STRAIN"}
		self.channel_dict = channel_dict_from_channel_list(options.channel_name)

		## A dictionary for shared memory partition, e.g., {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		self.shm_part_dict = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		if options.shared_memory_partition is not None:
			self.shm_part_dict.update( channel_dict_from_channel_list(options.shared_memory_partition) )

		## Seek event. Default is None, i.e., no seek
		self.seekevent = None

		## Analysis segment. Default is None
		self.seg = None

		if options.gps_start_time is not None:
			## Segment from gps start and stop time if given
			self.seg = segments.segment(LIGOTimeGPS(options.gps_start_time), LIGOTimeGPS(options.gps_end_time))
			## Seek event from the gps start and stop time if given
			self.seekevent = gst.event_new_seek(1., gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT, gst.SEEK_TYPE_SET, self.seg[0].ns(), gst.SEEK_TYPE_SET, self.seg[1].ns())

		if options.frame_segments_file is not None:
			## Frame segments from a user defined file
			self.frame_segments = ligolw_segments.segmenttable_get_by_name(utils.load_filename(options.frame_segments_file, contenthandler=ContentHandler), options.frame_segments_name).coalesce()
			if self.seg is not None:
				## Frame segments will be clipped to seek segment if it exists
				self.frame_segments = segments.segmentlistdict((instrument, seglist & segments.segmentlist([self.seg])) for instrument, seglist in self.frame_segments.items())
		else:
			## if no frame segments provided, set them to an empty segment list dictionary
			self.frame_segments = segments.segmentlistdict((instrument, None) for instrument in self.channel_dict)

		## DQ (state vector) channel dictionary, e.g., { "H1": "LLD-DQ_VECTOR", "H2": "LLD-DQ_VECTOR","L1": "LLD-DQ_VECTOR", "V1": "LLD-DQ_VECTOR" }
		self.dq_channel_dict = { "H1": "LLD-DQ_VECTOR", "H2": "LLD-DQ_VECTOR","L1": "LLD-DQ_VECTOR", "V1": "LLD-DQ_VECTOR" }

		## DQ channel type, e.g., "LLD"
		self.dq_channel_type = "LLD"

		if options.dq_channel_name is not None:
			dq_channel_dict_from_options = channel_dict_from_channel_list( options.dq_channel_name )
			instrument = dq_channel_dict_from_options.keys()[0]
			self.dq_channel_dict.update( dq_channel_dict_from_options )
			dq_channel = self.dq_channel_dict[instrument]
			if "ODC_" in dq_channel.split("-")[1]:
				self.dq_channel_type = "ODC"
	
		## Dictionary of state vector on, off bits like {"H1" : [0x7, 0x160], "H2" : [0x7, 0x160], "L1" : [0x7, 0x160], "V1" : [0x67, 0x100]}
		self.state_vector_on_off_bits = state_vector_on_off_dict_from_bit_lists(options.state_vector_on_bits, options.state_vector_off_bits)
		
		## frame cache file
		self.frame_cache = options.frame_cache
		## block size in bytes to read data from disk
		self.block_size = options.block_size
		## Data source, one of python.datasource.GWDataSourceInfo.data_sources
		self.data_source = options.data_source
		## Injection file name
		self.injection_filename = options.injections

		if options.data_source == "nds":
			## Store the ndssrc specific options: host
			self.nds_host = options.nds_host
			## Store the ndssrc specific options: port
			self.nds_port = options.nds_port
			## Store the ndssrc specific options: channel_type
			self.nds_channel_type = options.nds_channel_type

## #### Generic data source options used by many programs to append options to an OptionParser
# _Python doc string_:
def append_options(parser):
	"""!
	Append generic data source options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout the project
	for applications that read GW data.
	
-	--data-source [string]
		Set the data source from [frames|framexmitsrc|lvshm|nds|silence|white|AdvVirgo|LIGO|AdvLIGO].

-	--block-size [int] (bytes)
		Data block size to read in bytes. Default 16384 * 8 * 512 which is 512 seconds of double
		precision data at 16384 Hz. This parameter is only used if --data-source is one of
		white, silence, AdvVirgo, LIGO, AdvLIGO, nds.

-	--frame-cache [filename]
		Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).
		This is required iff --data-sourceframes)

-	--gps-start-time [int] (seconds)
		Set the start time of the segment to analyze in GPS seconds.
		Required unless --data-source in lvshm,framexmit

-	--gps-end-time  [int] (seconds)
		Set the end time of the segment to analyze in GPS seconds.  
		Required unless --data-source in lvshm,framexmit

-	--injections [filename]
		Set the name of the LIGO light-weight XML file from which to load injections (optional).

-	--channel-name [string]
		Set the name of the channels to process.
		Can be given multiple times as --channel-name=IFO=CHANNEL-NAME

-	--nds-host [hostname]
		Set the remote host or IP address that serves nds data.
		This is required iff --data-source is nds

-	--nds-port [portnumber]
		Set the port of the remote host that serves nds data, default = 31200.
		This is required iff --data-source is nds

-	--nds-channel-type [string] type
		FIXME please document

-	--dq-channel-name [string]
		Set the name of the data quality (or state vector) channel.
		This channel will be used to control the flow of data via the on/off bits.
		Can be given multiple times as --channel-nameIFOCHANNEL-NAME)

-	--shared-memory-partition [string]
		Set the name of the shared memory partition for a given instrument.
		Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME

-	--frame-segments-file [filename]
		Set the name of the LIGO light-weight XML file from which to load frame segments.
		Optional iff --data-source is frames

-	--frame-segments-name [string]
		Set the name of the segments to extract from the segment tables.
		Required iff --frame-segments-file is given

-	--state-vector-on-bits [hex]
		Set the state vector on bits to process (optional).
		The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.
		Only currently has meaning for online (lvshm, framexmit) data

-	--state-vector-off-bits [hex]
		Set the state vector off bits to process (optional).
		The default is 0x160 for all detectors. Override with IFO=bits can be given multiple times.
		Only currently has meaning for online (lvshm) data.)
	"""
	group = optparse.OptionGroup(parser, "Data source options", "Use these options to set up the appropriate data source")
	group.add_option("--data-source", metavar = "source", help = "Set the data source from [frames|framexmitsrc|lvshm|nds|silence|white|AdvVirgo|LIGO|AdvLIGO].  Required.")
	group.add_option("--block-size", type="int", metavar = "bytes", default = 16384 * 8 * 512, help = "Data block size to read in bytes. Default 16384 * 8 * 512 (512 seconds of double precision data at 16384 Hz.  This parameter is only used if --data-source is one of white, silence, AdvVirgo, LIGO, AdvLIGO, nds.")
	group.add_option("--frame-cache", metavar = "filename", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).  This is required iff --data-source=frames")
	group.add_option("--gps-start-time", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds. Required unless --data-source=lvshm")
	group.add_option("--gps-end-time", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds.  Required unless --data-source=lvshm")
	group.add_option("--injections", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load injections (optional).")
	group.add_option("--channel-name", metavar = "name", action = "append", help = "Set the name of the channels to process.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--nds-host", metavar = "hostname", help = "Set the remote host or IP address that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-port", metavar = "portnumber", type=int, default=31200, help = "Set the port of the remote host that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-channel-type", metavar = "type", default = "online", help = "Set the port of the remote host that serves nds data. This is required only if --data-source=nds. default==online")	
	group.add_option("--dq-channel-name", metavar = "name", action = "append", help = "Set the name of the data quality (or state vector) channel.  This channel will be used to control the flow of data via the on/off bits.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--shared-memory-partition", metavar = "name", action = "append", help = "Set the name of the shared memory partition for a given instrument.  Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME")
	group.add_option("--frame-segments-file", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load frame segments.  Optional iff --data-source=frames")
	group.add_option("--frame-segments-name", metavar = "name", help = "Set the name of the segments to extract from the segment tables.  Required iff --frame-segments-file is given")
	group.add_option("--state-vector-on-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector on bits to process (optional).  The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	group.add_option("--state-vector-off-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector off bits to process (optional).  The default is 0x160 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	parser.add_option_group(group)


## @cond DONTDOCUMENT
def do_seek(pipeline, seekevent):
	# FIXME:  remove.  seek the pipeline instead
	# DO NOT USE IN NEW CODE!!!!
	for src in pipeline.iterate_sources():
		if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
			raise RuntimeError("Element %s did not want to enter ready state" % src.get_name())
		if not src.send_event(seekevent):
			raise RuntimeError("Element %s did not handle seek event" % src.get_name())
## @endcond


## #### Gate controlled by a segment source
#
# ##### Gstreamer graph describing this function:
#
# @dot
# digraph G {
#	compound=true;
#	node [shape=record fontsize=10 fontname="Verdana"];
#	rankdir=LR;
# 	lal_gate;
#	lal_segmentsrc [URL="\ref python.pipeparts.mksegmentsrc()"];
#	lal_gate [URL="\ref python.pipeparts.mkgate()"];
#	in [label="?"];
#	out [label="?"];
#	in -> lal_gate -> out;
#	lal_segmentsrc -> lal_gate;
# }
# @enddot
# _Python doc string_:
def mksegmentsrcgate(pipeline, src, segment_list, seekevent = None, invert_output = False):
	"""!
	Takes a segment list and produces a gate driven by it. Hook up your own input and output.
	"""

	segsrc = pipeparts.mksegmentsrc(pipeline, segment_list, invert_output=invert_output)
	# FIXME:  remove
	if seekevent is not None:
		do_seek(pipeline, seekevent)
	return pipeparts.mkgate(pipeline, src, threshold = 1, control = segsrc)


## #### All-in-one data source
#
# ##### Gstreamer graph describing this function
#
# @dot
# digraph mkbasicsrc {
#      compound=true;
#      node [shape=record fontsize=10 fontname="Verdana"];
#      subgraph clusterfakesrc {
#              fake_0 [label="fakesrc: white, silence, AdvVirgo, LIGO, AdvLIGO", URL="\ref python.pipeparts.mkfakesrc()"];
#              color=black;
#              label="Possible path #1";
#      }
#      subgraph clusterframes {
#              color=black;
#              frames_0 [label="lalcachesrc: frames", URL="\ref python.pipeparts.mklalcachesrc()"];
#              frames_1 [label ="framecppchanneldemux", URL="\ref python.pipeparts.mkframecppchanneldemux()"];
#              frames_2 [label ="queue", URL="\ref python.pipeparts.mkqueue()"];
#              frames_3 [label ="gate (if user provides segments)", style=filled, color=lightgrey, URL="\ref python.pipeparts.mkgate()"];
#              frames_4 [label ="audiorate", URL="\ref python.pipeparts.mkaudiorate()"];
#              frames_0 -> frames_1 -> frames_2 -> frames_3 ->frames_4;
#              label="Possible path #2";
#      }
#	subgraph clusteronline {
#		color=black;
#		online_0 [label="lvshmsrc|framexmitsrc", URL="\ref python.pipeparts.mklvshmsrc()"];
#		online_1 [label ="framecppchanneldemux", URL="\ref python.pipeparts.mkframecppchanneldemux()"];
#		online_2a [label ="strain queue", URL="\ref python.pipeparts.mkqueue()"];
#		online_2b [label ="statevector queue", URL="\ref python.pipeparts.mkqueue()"];
#		online_3 [label ="statevector", URL="\ref python.pipeparts.mkstatevector()"];
#		online_4 [label ="gate", URL="\ref python.pipeparts.mkgate()"];
#		online_5 [label ="audiorate", URL="\ref python.pipeparts.mkaudiorate()"];
#		online_6 [label ="queue", URL="\ref python.pipeparts.mkqueue()"];
#		online_0 -> online_1;
#		online_1 -> online_2a;
#		online_1 -> online_2b;
#		online_2b -> online_3;
#		online_2a -> online_4;
#		online_3 -> online_4 -> online_5 -> online_6;
#		label="Possible path #3";
#	}
#	subgraph clusternds {
#		nds_0 [label="ndssrc", URL="\ref python.pipeparts.mkndssrc()"];
#		color=black;
#		label="Possible path #4";
#	}
#	audioconv [label="audioconvert", URL="\ref python.pipeparts.mkaudioconvert()"];
#	progress [label="progressreport (if verbose)", style=filled, color=lightgrey, URL="\ref python.pipeparts.mkprogressreport()"];
#	sim [label="lalsimulation (if injections requested)", style=filled, color=lightgrey, URL="\ref python.pipeparts.mkinjections()"];
#	queue [label="queue (if injections requested)", style=filled, color=lightgrey, URL="\ref python.pipeparts.mkqueue()"];
#
#	// The connections
#	fake_0 -> audioconv [ltail=clusterfakesrc];
#	frames_4 -> audioconv [ltail=clusterframes];
#	online_6 -> audioconv [ltail=clusteronline];
#	nds_0 -> audioconv [ltail=clusternds];
#	audioconv -> progress -> sim -> queue -> "?"
# }
# @enddot
# _Python doc string_:
def mkbasicsrc(pipeline, gw_data_source_info, instrument, verbose = False):
	"""!
	All the conditionals and stupid pet tricks for reading real or
	simulated h(t) data in one place.

	Consult the append_options() function and the GWDataSourceInfo class

	This src in general supports only one instrument although
	GWDataSourceInfo contains dictionaries of multi-instrument things.  By
	specifying the instrument when calling this function you will get ony a single
	instrument source.  A code wishing to have multiple basicsrcs will need to call
	this function for each instrument.
	"""

	if gw_data_source_info.data_source == "white":
		src = pipeparts.mkfakesrc(pipeline, instrument, gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, volume = 1.0)
	elif gw_data_source_info.data_source == "silence":
		src = pipeparts.mkfakesrc(pipeline, instrument, gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, wave = 4)
	elif gw_data_source_info.data_source == "LIGO":
		src = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "AdvLIGO":
		src = pipeparts.mkfakeadvLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "AdvVirgo":
		src = pipeparts.mkfakeadvvirgosrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "frames":
		if instrument == "V1":
			#FIXME Hack because virgo often just uses "V" in the file names rather than "V1".  We need to sieve on "V"
			src = pipeparts.mklalcachesrc(pipeline, location = gw_data_source_info.frame_cache, use_mmap = True, cache_src_regex = "V")
		else:
			src = pipeparts.mklalcachesrc(pipeline, location = gw_data_source_info.frame_cache, use_mmap = True, cache_src_regex = instrument[0], cache_dsc_regex = instrument)
		demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = True, channel_list = map("%s:%s".__mod__, gw_data_source_info.channel_dict.items()))
		pipeparts.framecpp_channeldemux_set_units(demux, dict.fromkeys(demux.get_property("channel-list"), "strain"))
		# allow frame reading and decoding to occur in a diffrent
		# thread
		src = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 8 * gst.SECOND)
		pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), src.get_pad("sink"))
		if gw_data_source_info.frame_segments[instrument] is not None:
			# FIXME:  make segmentsrc generate segment samples at the sample rate of h(t)?
			# FIXME:  make gate leaky when I'm certain that will work.
			src = pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mksegmentsrc(pipeline, gw_data_source_info.frame_segments[instrument]))
			pipeparts.framecpp_channeldemux_check_segments.set_probe(src.get_pad("src"), gw_data_source_info.frame_segments[instrument])
		# FIXME:  remove this when pipeline can handle disconts
		src = pipeparts.mkaudiorate(pipeline, src, skip_to_first = True, silent = False)
	elif gw_data_source_info.data_source in ("framexmit", "lvshm"):
		# See https://wiki.ligo.org/DAC/ER2DataDistributionPlan#LIGO_Online_DQ_Channel_Specifica
		state_vector_on_bits, state_vector_off_bits = gw_data_source_info.state_vector_on_off_bits[instrument]

		if gw_data_source_info.data_source == "lvshm":
			# FIXME make wait_time adjustable through web interface or command line or both
			src = pipeparts.mklvshmsrc(pipeline, shm_name = gw_data_source_info.shm_part_dict[instrument], wait_time = 120)
		elif gw_data_source_info.data_source == "framexmit":
			src = pipeparts.mkframexmitsrc(pipeline, multicast_group = framexmit_ports["CIT"][instrument][0], port = framexmit_ports["CIT"][instrument][1])
		else:
			# impossible code path
			raise ValueError(gw_data_source_info.data_source)

		src = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = True, skip_bad_files = True)
		pipeparts.framecpp_channeldemux_set_units(src, {"%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]): "strain"})

		# strain
		strain = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND * 60 * 1) # 1 minutes of buffering
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), strain.get_pad("sink"))
		# state vector
		# FIXME:  don't hard-code channel name
		statevector = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND * 60 * 1) # 1 minutes of buffering
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.dq_channel_dict[instrument]), statevector.get_pad("sink"))
		if gw_data_source_info.dq_channel_type == "ODC":
			# FIXME: This goes away when the ODC channel format is fixed.
			statevector = pipeparts.mkgeneric(pipeline, statevector, "lal_fixodc")
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
		src = pipeparts.mkgate(pipeline, strain, threshold = 1, control = statevector, default_state = False, name = "%s_state_vector_gate" % instrument)
		# export state vector state
		if gw_data_source_info.gate_start_callback is not None:
			src.set_property("emit-signals", True)
			src.connect("start", gw_data_source_info.gate_start_callback)
		if gw_data_source_info.gate_stop_callback is not None:
			src.set_property("emit-signals", True)
			src.connect("stop", gw_data_source_info.gate_stop_callback)
		src = pipeparts.mkaudiorate(pipeline, src, skip_to_first = True, silent = False)
		@bottle.route("/%s/strain_add_drop.txt" % instrument)
		def strain_add(elem = src):
			import time
			from pylal.date import XLALUTCToGPS
			t = float(XLALUTCToGPS(time.gmtime()))
			add = elem.get_property("add")
			drop = elem.get_property("drop")
			# FIXME don't hard code the sample rate
			return "%.9f %d %d" % (t, add / 16384., drop / 16384.)

		# 10 minutes of buffering
		src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND * 60 * 10)
	elif gw_data_source_info.data_source == "nds":
		src = pipeparts.mkndssrc(pipeline, gw_data_source_info.nds_host, instrument, gw_data_source_info.channel_dict[instrument], gw_data_source_info.nds_channel_type, blocksize = gw_data_source_info.block_size, port = gw_data_source_info.nds_port)
	else:
		raise ValueError("invalid data_source: %s" % gw_data_source_info.data_source)

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
		# let the injection code run in a different thread than the
		# whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = gst.SECOND * 64)

	#
	# seek the pipeline
	# FIXME:  remove
	#

	if gw_data_source_info.data_source in ("white", "silence", "LIGO", "AdvLIGO", "AdvVirgo", "frames"):
		do_seek(pipeline, gw_data_source_info.seekevent)

	#
	# done
	#

	return src


## #### h(t) Gate: quick glitch excision trick
#
# ##### Gstreamer graph
#
# @dot
# digraph G {
#	compound=true;
#	node [shape=record fontsize=10 fontname="Verdana"];
#	rankdir=LR;
# 	tee [URL="\ref python.pipeparts.mktee()"];
# 	inputqueue [URL="\ref python.pipeparts.mkqueue()"];
# 	controlqueue [URL="\ref python.pipeparts.mkqueue()"];
#	lal_gate [URL="\ref python.pipeparts.mkgate()"];
#	in [label="?"];
#	out [label="?"];
#	in -> tee -> inputqueue -> lal_gate -> out;
#	tee -> controlqueue -> lal_gate;
# }
# @enddot
# _Python doc string_:
def mkhtgate(pipeline, src, control = None, threshold = 8.0, attack_length = -128, hold_length = -128, name = None):
	"""!
	A convenience function to provide thresholds on input data.  This can
	be used to remove large spikes / glitches etc.  Of course you can use it for
	other stuff by plugging whatever you want as input and ouput
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
