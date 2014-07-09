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
# ### Review Status
#
# | Names                                       | Hash                                        | Date       |
# | ------------------------------------------- | ------------------------------------------- | ---------- |
# | Florent, Sathya, Duncan Me., Jolien, Kipp, Chad | b3ef077fe87b597578000f140e4aa780f3a227aa    | 2014-05-01 |
#
# #### Action items
#
# - State vector, DQ, etc must be in place by November and should be coordinated with JRPC/DAC/DetChar/CBC/etc
# - Consider a dynamic data time out that seneses when data is not going to arrive for a while

##
# @package python.datasource
#
# datasource module
#


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


def channel_dict_from_channel_list(channel_list):
	"""!
	Given a list of channels, produce a dictionary keyed by ifo of channel names:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> channel_dict_from_channel_list(["H1=LSC-STRAIN", "H2=SOMETHING-ELSE"])
		{'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}
	"""
	return dict(instrument_channel.split("=") for instrument_channel in channel_list)


def pipeline_channel_list_from_channel_dict(channel_dict, ifos = None, opt = "channel-name"):
	"""!
	Creates a string of channel names options from a dictionary keyed by ifos.

	FIXME: This function exists to work around pipeline.py's inability to
	give the same option more than once by producing a string to pass as an argument
	that encodes the other instances of the option.

	- override --channel-name with a different option by setting opt.
	- restrict the ifo keys to a subset of the channel_dict by 
	  setting ifos

	Examples:

		>>> pipeline_channel_list_from_channel_dict({'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'})
		'H2=SOMETHING-ELSE --channel-name=H1=LSC-STRAIN '

		>>> pipeline_channel_list_from_channel_dict({'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}, ifos=["H1"])
		'H1=LSC-STRAIN '

		>>> pipeline_channel_list_from_channel_dict({'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}, opt="test-string")
		'H2=SOMETHING-ELSE --test-string=H1=LSC-STRAIN '
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


def state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list, state_vector_on_off_dict = state_vector_on_off_dict):
	"""!
	Produce a dictionary (keyed by detector) of on / off bit tuples from a
	list provided on the command line.

	Takes default values from module level datasource.state_vector_on_off_dict
	if state_vector_on_off_dict is not given

	Inputs must be given as base 10 or 16 integers

	Examples:

		>>> on_bit_list = ["V1=7", "H1=7", "L1=7"]
		>>> off_bit_list  = ["V1=256", "H1=352", "L1=352"]
		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list)
		{'H2': [7, 352], 'V1': [7, 256], 'H1': [7, 352], 'L1': [7, 352]}

		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list,{})
		{'V1': [7, 256], 'H1': [7, 352], 'L1': [7, 352]}

		>>> on_bit_list = ["V1=0x7", "H1=0x7", "L1=0x7"]
		>>> off_bit_list = ["V1=0x256", "H1=0x352", "L1=0x352"]
		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list,{})
		{'V1': [7, 598], 'H1': [7, 850], 'L1': [7, 850]}
	"""
	for line in on_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		try:
			val = int(bits)
		except ValueError: # could be hex, that is all we support other than int
			val = int(bits, 16)
		try:
			state_vector_on_off_dict[ifo][0] = val
		except KeyError:
			state_vector_on_off_dict[ifo] = [val, 0]
	
	for line in off_bit_list:
		ifo = line.split("=")[0]
		bits = "".join(line.split("=")[1:])
		# shouldn't have to worry about key errors at this point
		try:
			state_vector_on_off_dict[ifo][1] = int(bits)
		except ValueError: # must be hex
			state_vector_on_off_dict[ifo][1] = int(bits, 16)

	return state_vector_on_off_dict


def state_vector_on_off_list_from_bits_dict(bit_dict):
	"""!
	Produce a tuple of useful command lines from a dictionary of on / off state
	vector bits keyed by detector
	
	FIXME: This function exists to work around pipeline.py's inability to
	give the same option more than once by producing a string to pass as an argument
	that encodes the other instances of the option.

	Examples:

		>>> state_vector_on_off_dict = {"H1":[0x7, 0x160], "H2":[0x7, 0x160], "L1":[0x7, 0x160], "V1":[0x67, 0x100]}
		>>> state_vector_on_off_list_from_bits_dict(state_vector_on_off_dict)
		('H2=7 --state-vector-on-bits=V1=103 --state-vector-on-bits=H1=7 --state-vector-on-bits=L1=7 ', 'H2=352 --state-vector-off-bits=V1=256 --state-vector-off-bits=H1=352 --state-vector-off-bits=L1=352 ')
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


def seek_event_for_gps(gps_start_time, gps_end_time, flags = 0):
	"""!
	Create a new seek event, i.e., gst.event_new_seek()  for a given
	gps_start_time and gps_end_time, with optional flags.  

	@param gps_start_time start time as LIGOTimeGPS, double or float
	@param gps_end_time start time as LIGOTimeGPS, double or float
	"""
	def seek_args_for_gps(gps_time):
		"""!
		Convenience routine to convert a GPS time to a seek type and a
		GStreamer timestamp.
		"""

		if gps_time is None or gps_time == -1:
			return (gst.SEEK_TYPE_NONE, -1) # -1 == gst.CLOCK_TIME_NONE
		elif hasattr(gps_time, 'ns'):
			return (gst.SEEK_TYPE_SET, gps_time.ns())
		else:
			return (gst.SEEK_TYPE_SET, long(float(gps_time) * gst.SECOND))

	start_type, start_time = seek_args_for_gps(gps_start_time)
	stop_type, stop_time   = seek_args_for_gps(gps_end_time)

	return gst.event_new_seek(1., gst.FORMAT_TIME, flags, start_type, start_time, stop_type, stop_time)


class GWDataSourceInfo(object):
	"""!
	Hold the data associated with data source command lines.
	"""
	## See datasource.append_options()
	def __init__(self, options):
		"""!
		Initialize a GWDataSourceInfo class instance from command line options specified by append_options()
		""" 

		## A list of possible, valid data sources ("frames", "framexmit", "lvshm", "nds", "white", "silence", "AdvVirgo", "LIGO", "AdvLIGO")
		self.data_sources = set(("frames", "framexmit", "lvshm", "nds", "white", "silence", "AdvVirgo", "LIGO", "AdvLIGO"))
		self.live_sources = set(("framexmit", "lvshm"))
		assert self.live_sources <= self.data_sources

		# Sanity check the options
		if options.data_source not in self.data_sources:
			raise ValueError("--data-source must be one of %s" % ", ".join(self.data_sources))
		if options.data_source == "frames" and options.frame_cache is None:
			raise ValueError("--frame-cache must be specified when using --data-source=frames")
		if not options.channel_name:
			raise ValueError("must specify at least one channel in the form --channel-name=IFO=CHANNEL-NAME")
		if options.frame_segments_file is not None and options.data_source != "frames":
			raise ValueError("can only give --frame-segments-file if --data-source=frames")
		if options.frame_segments_name is not None and options.frame_segments_file is None:
			raise ValueError("can only specify --frame-segments-name if --frame-segments-file is given")
		if options.data_source == "nds" and (options.nds_host is None or options.nds_port is None):
			raise ValueError("must specify --nds-host and --nds-port when using --data-source=nds")

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
			## Seek event from the gps start and stop time if given
			self.seekevent = gst.event_new_seek(1., gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT, gst.SEEK_TYPE_SET, self.seg[0].ns(), gst.SEEK_TYPE_SET, self.seg[1].ns())
		elif options.gps_end_time is not None:
			raise ValueError("must provide both --gps-start-time and --gps-end-time")
		elif options.data_source not in self.live_sources:
			raise ValueError("--gps-start-time and --gps-end-time must be specified when %s" % " or ".join("--data-source=%s" % src for src in sorted(self.live_sources)))

		if options.frame_segments_file is not None:
			## Frame segments from a user defined file
			self.frame_segments = ligolw_segments.segmenttable_get_by_name(utils.load_filename(options.frame_segments_file, contenthandler=ContentHandler), options.frame_segments_name).coalesce()
			if self.seg is not None:
				# Clip frame segments to seek segment if it
				# exists (not required, just saves some
				# memory and I/O overhead)
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

	#### Typical usage case examples

	-# Reading data from frames

		--data-source=frames --gps-start-time=999999000 --gps-end-time=999999999 --channel-name=H1=LDAS-STRAIN --frame-segments-file=segs.xml --frame-segments-name=datasegments

	-# Reading data from a fake LIGO source
		
		--data-source=LIGO --gps-start-time=999999000 --gps-end-time=999999999 --channel-name=H1=FAIKE-STRAIN

	-# Reading online data via framexmit

		--data-source=framexmit --channel-name=H1=FAIKE-STRAIN

	-# Many other combinations possible, please add some!
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


##
# _Gstreamer graph describing this function:_
#
# @dot
# digraph G {
#	compound=true;
#	node [shape=record fontsize=10 fontname="Verdana"];
#	rankdir=LR;
# 	lal_gate;
#	lal_segmentsrc [URL="\ref pipeparts.mksegmentsrc()"];
#	lal_gate [URL="\ref pipeparts.mkgate()"];
#	in [label="?"];
#	out [label="?"];
#	in -> lal_gate -> out;
#	lal_segmentsrc -> lal_gate;
# }
# @enddot
#
#
def mksegmentsrcgate(pipeline, src, segment_list, seekevent = None, invert_output = False, **kwargs):
	"""!
	Takes a segment list and produces a gate driven by it. Hook up your own input and output.

	@param kwargs passed through to pipeparts.mkgate(), e.g., used to set the gate's name.
	"""
	segsrc = pipeparts.mksegmentsrc(pipeline, segment_list, invert_output = invert_output)
	# FIXME:  remove
	if seekevent is not None:
		do_seek(pipeline, seekevent)
	return pipeparts.mkgate(pipeline, src, threshold = 1, control = segsrc, **kwargs)


##
# _Gstreamer graph describing this function:_
#
# @dot
# digraph mkbasicsrc {
#      compound=true;
#      node [shape=record fontsize=10 fontname="Verdana"];
#      subgraph clusterfakesrc {
#              fake_0 [label="fakesrc: white, silence, AdvVirgo, LIGO, AdvLIGO", URL="\ref pipeparts.mkfakesrc()"];
#              color=black;
#              label="Possible path #1";
#      }
#      subgraph clusterframes {
#              color=black;
#              frames_0 [label="lalcachesrc: frames", URL="\ref pipeparts.mklalcachesrc()"];
#              frames_1 [label ="framecppchanneldemux", URL="\ref pipeparts.mkframecppchanneldemux()"];
#              frames_2 [label ="queue", URL="\ref pipeparts.mkqueue()"];
#              frames_3 [label ="gate (if user provides segments)", style=filled, color=lightgrey, URL="\ref pipeparts.mkgate()"];
#              frames_4 [label ="audiorate", URL="\ref pipeparts.mkaudiorate()"];
#              frames_0 -> frames_1 -> frames_2 -> frames_3 ->frames_4;
#              label="Possible path #2";
#      }
#	subgraph clusteronline {
#		color=black;
#		online_0 [label="lvshmsrc|framexmitsrc", URL="\ref pipeparts.mklvshmsrc()"];
#		online_1 [label ="framecppchanneldemux", URL="\ref pipeparts.mkframecppchanneldemux()"];
#		online_2a [label ="strain queue", URL="\ref pipeparts.mkqueue()"];
#		online_2b [label ="statevector queue", URL="\ref pipeparts.mkqueue()"];
#		online_3 [label ="statevector", URL="\ref pipeparts.mkstatevector()"];
#		online_4 [label ="gate", URL="\ref pipeparts.mkgate()"];
#		online_5 [label ="audiorate", URL="\ref pipeparts.mkaudiorate()"];
#		online_6 [label ="queue", URL="\ref pipeparts.mkqueue()"];
#		online_0 -> online_1;
#		online_1 -> online_2a;
#		online_1 -> online_2b;
#		online_2b -> online_3;
#		online_2a -> online_4;
#		online_3 -> online_4 -> online_5 -> online_6;
#		label="Possible path #3";
#	}
#	subgraph clusternds {
#		nds_0 [label="ndssrc", URL="\ref pipeparts.mkndssrc()"];
#		color=black;
#		label="Possible path #4";
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
			src = pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mksegmentsrc(pipeline, gw_data_source_info.frame_segments[instrument]), name = "%s_frame_segments_gate" % instrument)
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
			src = pipeparts.mkframexmitsrc(pipeline, multicast_group = framexmit_ports["CIT"][instrument][0], port = framexmit_ports["CIT"][instrument][1], wait_time = 120)
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

		# fill in holes, skip duplicate data
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


## 
# _Gstreamer graph describing the pipeline_
#
# @dot
# digraph G {
#	compound=true;
#	node [shape=record fontsize=10 fontname="Verdana"];
#	rankdir=LR;
# 	tee [URL="\ref pipeparts.mktee()"];
# 	inputqueue [URL="\ref pipeparts.mkqueue()"];
#	lal_gate [URL="\ref pipeparts.mkgate()"];
#	in [label="?"];
#	out [label="?"];
#	in -> tee -> inputqueue -> lal_gate -> out;
#	tee -> lal_gate;
# }
# @enddot
#
#
def mkhtgate(pipeline, src, control = None, threshold = 8.0, attack_length = 128, hold_length = 128, **kwargs):
	"""!
	A convenience function to provide thresholds on input data.  This can
	be used to remove large spikes / glitches etc.  Of course you can use it for
	other stuff by plugging whatever you want as input and ouput

	NOTE:  the queues constructed by this code assume the attack and
	hold lengths combined are less than 1 second in duration.
	"""
	# FIXME someday explore a good bandpass filter
	# src = pipeparts.mkaudiochebband(pipeline, src, low_frequency, high_frequency)
	if control is None:
		control = src = pipeparts.mktee(pipeline, src)
	src = pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND, max_size_bytes = 0, max_size_buffers = 0)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = threshold, attack_length = -attack_length, hold_length = -hold_length, invert_control = True, **kwargs)

# Unit tests
if __name__ == "__main__":
	import doctest
	doctest.testmod()
