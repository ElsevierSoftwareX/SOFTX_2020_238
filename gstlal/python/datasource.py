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

__doc__ = """

A file that contains the datasource module code

*Review Status*

+-------------------------------------------------+------------------------------------------+------------+
| Names                                           | Hash                                     | Date       |
+=================================================+==========================================+============+
| Florent, Sathya, Duncan Me., Jolien, Kipp, Chad | b3ef077fe87b597578000f140e4aa780f3a227aa | 2014-05-01 |
+-------------------------------------------------+------------------------------------------+------------+

"""


import optparse
import sys
import tempfile
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS
from ligo import segments
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import segments as ligolw_segments

from gstlal import bottle
from gstlal import pipeparts
from gstlal.dags import util as dagutil


#
# Misc useful functions
#


def channel_dict_from_channel_list(channel_list):
	"""
	Given a list of channels, produce a dictionary keyed by ifo of channel names:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> channel_dict_from_channel_list(["H1=LSC-STRAIN", "H2=SOMETHING-ELSE"])
		{'H1': 'LSC-STRAIN', 'H2': 'SOMETHING-ELSE'}
	"""
	return dict(instrument_channel.split("=") for instrument_channel in channel_list)

def channel_dict_from_channel_list_with_node_range(channel_list):
	"""
	Given a list of channels with a range of mass bins, produce a dictionary
	keyed by ifo of channel names:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> channel_dict_from_channel_list_with_node_range(["0000:0002:H1=LSC_STRAIN_1,L1=LSC_STRAIN_2", "0002:0004:H1=LSC_STRAIN_3,L1=LSC_STRAIN_4", "0004:0006:H1=LSC_STRAIN_5,L1=LSC_STRAIN_6"])
		{'0000': {'H1': 'LSC_STRAIN_1', 'L1': 'LSC_STRAIN_2'}, '0001': {'H1': 'LSC_STRAIN_1', 'L1': 'LSC_STRAIN_2'}, '0002': {'H1': 'LSC_STRAIN_3', 'L1': 'LSC_STRAIN_4'}, '0003': {'H1': 'LSC_STRAIN_3', 'L1': 'LSC_STRAIN_4'}, '0004': {'H1': 'LSC_STRAIN_5', 'L1': 'LSC_STRAIN_6'}, '0005': {'H1': 'LSC_STRAIN_5', 'L1': 'LSC_STRAIN_6'}}
	"""
	outdict = {}
	for instrument_channel_full in channel_list:
		instrument_channel_split = instrument_channel_full.split(':')
		for ii in range(int(instrument_channel_split[0]),int(instrument_channel_split[1])):
			outdict[str(ii).zfill(4)] = dict((instrument_channel.split("=")) for instrument_channel in instrument_channel_split[2].split(','))
	return outdict

def pipeline_channel_list_from_channel_dict(channel_dict, ifos = None, opt = "channel-name"):
	"""
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

def pipeline_channel_list_from_channel_dict_with_node_range(channel_dict, node = 0, ifos = None, opt = "channel-name"):
	"""
	Creates a string of channel names options from a dictionary keyed by ifos.

	FIXME: This function exists to work around pipeline.py's inability to
	give the same option more than once by producing a string to pass as an argument
	that encodes the other instances of the option.

	- override --channel-name with a different option by setting opt.
	- restrict the ifo keys to a subset of the channel_dict by.
	  setting ifos

	Examples:

		>>> pipeline_channel_list_from_channel_dict_with_node_range({'0000': {'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}}, node=0)
		'H2=SOMETHING-ELSE --channel-name=H1=LSC-STRAIN '

		>>> pipeline_channel_list_from_channel_dict_with_node_range({'0000': {'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}}, node=0, ifos=["H1"])
		'H1=LSC-STRAIN '

		>>> pipeline_channel_list_from_channel_dict_with_node_range({'0000': {'H2': 'SOMETHING-ELSE', 'H1': 'LSC-STRAIN'}}, node=0, opt="test-string")
		'H2=SOMETHING-ELSE --test-string=H1=LSC-STRAIN '
	"""
	outstr = ""
	node = str(node).zfill(4)
	if ifos is None:
		ifos = channel_dict[node].keys()
	for i, ifo in enumerate(ifos):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[node][ifo])
		else:
			outstr += "--%s=%s=%s " % (opt, ifo, channel_dict[node][ifo])

	return outstr

def injection_dict_from_channel_list_with_node_range(injection_list):
	"""
	Given a list of injection xml files with a range of mass bins, produce a
	dictionary keyed by bin number:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:
	>>> injection_dict_from_channel_list_with_node_range(["0000:0002:Injection_1.xml", "0002:0004:Injection_2.xml"])
	{'0000': 'Injection_1.xml', '0001': 'Injection_1.xml', '0002': 'Injection_2.xml', '0003': 'Injection_2.xml'}
	"""
	outdict = {}
	for injection_name in injection_list:
		injection_name_split = injection_name.split(':')
		for ii in range(int(injection_name_split[0]),int(injection_name_split[1])):
			outdict[str(ii).zfill(4)] = injection_name_split[2]
	return outdict

## #### Default dictionary of state vector on/off bits by ifo
# Used as the default argument to state_vector_on_off_dict_from_bit_lists()
state_vector_on_off_dict = {
	"H1" : [0x7, 0x160],
	"H2" : [0x7, 0x160],
	"L1" : [0x7, 0x160],
	"V1" : [0x67, 0x100]
}


## #### Default dictionary of DQ vector on/off bits by ifo
# Used as the default argument to dq_vector_on_off_dict_from_bit_lists()
dq_vector_on_off_dict = {
	"H1" : [0x7, 0x0],
	"H2" : [0x7, 0x0],
	"L1" : [0x7, 0x0],
	"V1" : [0x7, 0x0]
}


def state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list, state_vector_on_off_dict = state_vector_on_off_dict):
	"""
	Produce a dictionary (keyed by detector) of on / off bit tuples from a
	list provided on the command line.

	Takes default values from module level datasource.state_vector_on_off_dict
	if state_vector_on_off_dict is not given

	Inputs must be given as base 10 or 16 integers

	Examples:

		>>> on_bit_list = ["V1=7", "H1=7", "L1=7"]
		>>> off_bit_list  = ["V1=256", "H1=352", "L1=352"]
		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list)
		{'H1': [7, 352], 'H2': [7, 352], 'L1': [7, 352], 'V1': [7, 256]}

		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list,{})
		{'V1': [7, 256], 'H1': [7, 352], 'L1': [7, 352]}

		>>> on_bit_list = ["V1=0x7", "H1=0x7", "L1=0x7"]
		>>> off_bit_list = ["V1=0x256", "H1=0x352", "L1=0x352"]
		>>> state_vector_on_off_dict_from_bit_lists(on_bit_list, off_bit_list,{})
		{'V1': [7, 598], 'H1': [7, 850], 'L1': [7, 850]}
	"""
	for ifo, bits in [line.strip().split("=", 1) for line in on_bit_list]:
		bits = int(bits, 16) if bits.startswith("0x") else int(bits)
		try:
			state_vector_on_off_dict[ifo][0] = bits
		except KeyError:
			state_vector_on_off_dict[ifo] = [bits, 0]

	for ifo, bits in [line.strip().split("=", 1) for line in off_bit_list]:
		bits = int(bits, 16) if bits.startswith("0x") else int(bits)
		# shouldn't have to worry about key errors at this point
		state_vector_on_off_dict[ifo][1] = bits

	return state_vector_on_off_dict


def state_vector_on_off_list_from_bits_dict(bit_dict):
	"""
	Produce a tuple of useful command lines from a dictionary of on / off state
	vector bits keyed by detector

	FIXME: This function exists to work around pipeline.py's inability to
	give the same option more than once by producing a string to pass as an argument
	that encodes the other instances of the option.

	Examples:

		>>> state_vector_on_off_dict = {"H1":[0x7, 0x160], "H2":[0x7, 0x160], "L1":[0x7, 0x160], "V1":[0x67, 0x100]}
		>>> state_vector_on_off_list_from_bits_dict(state_vector_on_off_dict)
		('H1=7 --state-vector-on-bits=H2=7 --state-vector-on-bits=L1=7 --state-vector-on-bits=V1=103 ', 'H1=352 --state-vector-off-bits=H2=352 --state-vector-off-bits=L1=352 --state-vector-off-bits=V1=256 ')
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


def framexmit_dict_from_framexmit_list(framexmit_list):
	"""
	Given a list of framexmit addresses with ports, produce a dictionary keyed by ifo:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> framexmit_dict_from_framexmit_list(["H1=224.3.2.1:7096", "L1=224.3.2.2:7097", "V1=224.3.2.3:7098"])
		{'H1': ('224.3.2.1', 7096), 'L1': ('224.3.2.2', 7097), 'V1': ('224.3.2.3', 7098)}
	"""
	out = []
	for instrument_addr in framexmit_list:
		ifo, addr_port = instrument_addr.split("=")
		addr, port = addr_port.split(':')
		out.append((ifo, (addr, int(port))))
	return dict(out)


def framexmit_list_from_framexmit_dict(framexmit_dict, ifos = None, opt = "framexmit-addr"):
	"""
	Creates a string of framexmit address options from a dictionary keyed by ifos.

	Examples:

		>>> framexmit_list_from_framexmit_dict({'V1': ('224.3.2.3', 7098), 'H1': ('224.3.2.1', 7096), 'L1': ('224.3.2.2', 7097)})
		'V1=224.3.2.3:7098 --framexmit-addr=H1=224.3.2.1:7096 --framexmit-addr=L1=224.3.2.2:7097 '
	"""
	outstr = ""
	if ifos is None:
		ifos = framexmit_dict.keys()
	for i, ifo in enumerate(ifos):
		if i == 0:
			outstr += "%s=%s:%s " % (ifo, framexmit_dict[ifo][0], framexmit_dict[ifo][1])
		else:
			outstr += "--%s=%s=%s:%s " % (opt, ifo, framexmit_dict[ifo][0], framexmit_dict[ifo][1])

	return outstr


def frame_type_dict_from_frame_type_list(frame_type_list):
	"""
	Given a list of frame types, produce a dictionary keyed by ifo:

	The list here typically comes from an option parser with options that
	specify the "append" action.

	Examples:

		>>> frame_type_dict_from_frame_type_list(['H1=H1_GWOSC_O2_16KHZ_R1', 'L1=L1_GWOSC_O2_16KHZ_R1'])
		{'H1': 'H1_GWOSC_O2_16KHZ_R1', 'L1': 'L1_GWOSC_O2_16KHZ_R1'}
	"""
	out = {}
	for frame_opt in frame_type_list:
		ifo, frame_type = frame_opt.split("=")
		out[ifo] = frame_type

	return out


def pipeline_seek_for_gps(pipeline, gps_start_time, gps_end_time, flags = Gst.SeekFlags.FLUSH):
	"""
	Create a new seek event, i.e., Gst.Event.new_seek()  for a given
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
			return (Gst.SeekType.NONE, -1) # -1 == Gst.CLOCK_TIME_NONE
		elif hasattr(gps_time, 'ns'):
			return (Gst.SeekType.SET, gps_time.ns())
		else:
			return (Gst.SeekType.SET, int(float(gps_time) * Gst.SECOND))

	start_type, start_time = seek_args_for_gps(gps_start_time)
	stop_type, stop_time   = seek_args_for_gps(gps_end_time)

	# FIXME:  should seek whole pipeline, but there are several
	# problems preventing us from doing that.
	#
	# because the framecpp demuxer has no source pads until decoding
	# begins, the bottom halves of pipelines start out disconnected
	# from the top halves of pipelines, which means the seek events
	# (which are sent to sink elements) don't make it all the way to
	# the source elements.  dynamic pipeline building will not fix the
	# problem because the dumxer does not carry the "SINK" flag so even
	# though it starts with only a sink pad and no source pads it still
	# won't be sent the seek event.  gstreamer's own demuxers must
	# somehow have a solution to this problem, but I don't know what it
	# is.  I notice that many implement the send_event() method
	# override, and it's possible that's part of the solution.
	#
	# seeking the pipeline can only be done in the PAUSED state.  the
	# GstBaseSrc baseclass seeks itself to 0 when changing to the
	# paused state, and the preroll is performed before the seek event
	# we send to the pipeline is processed, so the preroll occurs with
	# whatever random data a seek to "0" causes source elements to
	# produce.  for us, when processing GW data, this leads to the
	# whitener element's initial spectrum estimate being initialized
	# from that random data, and a non-zero chance of even getting
	# triggers out of it, all of which is very bad.
	#
	# the only way we have at the moment to solve both problems --- to
	# ensure seek events arrive at source elements and to work around
	# GstBaseSrc's initial seek to 0 --- is to send seek events
	# directly to the source elements ourselves before putting the
	# pipeline into the PAUSED state.  the elements are happy to
	# receive seek events in the READY state, and GstBaseSrc updtes its
	# current segment using that seek so that when it transitions to
	# the PAUSED state and does its intitial seek it seeks to our
	# requested time, not to 0.
	#
	# So:  this function needs to be called with the pipeline in the
	# READY state in order to guarantee the data stream starts at the
	# requested start time, and does not get prerolled with random
	# data.  For safety we include a check of the pipeline's current
	# state.
	#
	# if in the future we find some other solution to these problems
	# the story might change and the pipeline state required on entry
	# into this function might change.

	#pipeline.seek(1.0, Gst.Format(Gst.Format.TIME), flags, start_type, start_time, stop_type, stop_time)

	if pipeline.current_state != Gst.State.READY:
		raise ValueError("pipeline must be in READY state")

	for elem in pipeline.iterate_sources():
		elem.seek(1.0, Gst.Format(Gst.Format.TIME), flags, start_type, start_time, stop_type, stop_time)


class GWDataSourceInfo(object):
	"""
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
		if options.data_source == "frames" and options.frame_cache is None and options.frame_type is None:
			raise ValueError("--frame-cache or --frame-type must be specified when using --data-source=frames")
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
		self.shm_part_dict = {"H1": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}
		if options.shared_memory_partition is not None:
			self.shm_part_dict.update( channel_dict_from_channel_list(options.shared_memory_partition) )

		## options for shared memory
		self.shm_assumed_duration = options.shared_memory_assumed_duration
		self.shm_block_size = options.shared_memory_block_size # NOTE: should this be incorporated into options.block_size? currently only used for offline data sources

		## A dictionary of framexmit addresses
		self.framexmit_addr = framexmit_ports["CIT"]
		if options.framexmit_addr is not None:
			self.framexmit_addr.update( framexmit_dict_from_framexmit_list(options.framexmit_addr) )
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
			self.frame_segments = segments.segmentlistdict((instrument, None) for instrument in self.channel_dict)

		## DQ and state vector channel dictionary, e.g., { "H1": "LLD-DQ_VECTOR", "H2": "LLD-DQ_VECTOR","L1": "LLD-DQ_VECTOR", "V1": "LLD-DQ_VECTOR" }
		self.state_channel_dict = { "H1": "LLD-DQ_VECTOR", "H2": "LLD-DQ_VECTOR","L1": "LLD-DQ_VECTOR", "V1": "LLD-DQ_VECTOR" }
		self.dq_channel_dict = { "H1": "DMT-DQ_VECTOR", "H2": "DMT-DQ_VECTOR","L1": "DMT-DQ_VECTOR", "V1": "DMT-DQ_VECTOR" }

		if options.state_channel_name is not None:
			state_channel_dict_from_options = channel_dict_from_channel_list( options.state_channel_name )
			instrument = list(state_channel_dict_from_options.keys())[0]
			self.state_channel_dict.update( state_channel_dict_from_options )

		if options.dq_channel_name is not None:
			dq_channel_dict_from_options = channel_dict_from_channel_list( options.dq_channel_name )
			instrument = list(dq_channel_dict_from_options.keys())[0]
			self.dq_channel_dict.update( dq_channel_dict_from_options )

		## Dictionary of state vector on, off bits like {"H1" : [0x7, 0x160], "H2" : [0x7, 0x160], "L1" : [0x7, 0x160], "V1" : [0x67, 0x100]}
		self.state_vector_on_off_bits = state_vector_on_off_dict_from_bit_lists(options.state_vector_on_bits, options.state_vector_off_bits, state_vector_on_off_dict)
		self.dq_vector_on_off_bits = state_vector_on_off_dict_from_bit_lists(options.dq_vector_on_bits, options.dq_vector_off_bits, dq_vector_on_off_dict)

		## load frame cache
		if options.frame_cache is not None:
			self.frame_cache = options.frame_cache
		else:
			frame_type_dict = frame_type_dict_from_frame_type_list(options.frame_type)
			frame_cache = datafind.load_frame_cache(start, end, frame_type_dict, host=options.data_find_server)
			## create a temporary cache file
			self._frame_cache_fileobj = tempfile.NamedTemporaryFile(suffix=".cache", dir=dagutil.condor_scratch_space())
			self.frame_cache = self._frame_cache_fileobj.name
			with open(self.frame_cache, "w") as f:
				for cacheentry in frame_cache:
					print(str(cacheentry), file=f)

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
	"""
	Append generic data source options to an OptionParser object in order
	to have consistent an unified command lines and parsing throughout the project
	for applications that read GW data.

-	--data-source [string]
		Set the data source from [frames|framexmit|lvshm|nds|silence|white].

-	--block-size [int] (bytes)
		Data block size to read in bytes. Default 16384 * 8 * 512 which is 512 seconds of double
		precision data at 16384 Hz. This parameter is only used if --data-source is one of
		white, silence, AdvVirgo, LIGO, AdvLIGO, nds.

-	--frame-cache [filename]
		Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).

-	--frame-type [string]
		Set the frame type for a given instrument.
		Can be given multiple times as --frame-type=IFO=FRAME-TYPE

-	--gps-start-time [int] (seconds)
		Set the start time of the segment to analyze in GPS seconds.
		Required unless --data-source is lvshm or framexmit

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

-	--framexmit-addr [string]
		Set the address of the framexmit service.  Can be given
		multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port

-	--framexmit-iface [string]
		Set the address of the framexmit interface.

-	--state-channel-name [string]
		Set the name of the state vector channel.
		This channel will be used to control the flow of data via the on/off bits.
		Can be given multiple times as --state-channel-name=IFO=STATE-CHANNEL-NAME

-	--dq-channel-name [string]
		Set the name of the data quality channel.
		This channel will be used to control the flow of data via the on/off bits.
		Can be given multiple times as --state-channel-name=IFO=DQ-CHANNEL-NAME

-	--shared-memory-partition [string]
		Set the name of the shared memory partition for a given instrument.
		Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME

-	--shared-memory-assumed-duration [int]
		Set the assumed span of files in seconds. Default = 4 seconds.

-	--shared-memory-block-size [int]
		Set the byte size to read per buffer. Default = 4096 bytes.

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
		Only currently has meaning for online (lvshm, framexmit) data

-	--dq-vector-on-bits [hex]
		Set the state vector on bits to process (optional).
		The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.
		Only currently has meaning for online (lvshm, framexmit) data

-	--dq-vector-off-bits [hex]
		Set the dq vector off bits to process (optional).
		The default is 0x0 for all detectors. Override with IFO=bits can be given multiple times.
		Only currently has meaning for online (lvshm, framexmit) data

	**Typical usage case examples**

	1. Reading data from frames::

		--data-source=frames --gps-start-time=999999000 --gps-end-time=999999999 \\
		--channel-name=H1=LDAS-STRAIN --frame-segments-file=segs.xml \\
		--frame-segments-name=datasegments

	2. Reading data from a fake LIGO source::

		--data-source=LIGO --gps-start-time=999999000 --gps-end-time=999999999 \\
		--channel-name=H1=FAIKE-STRAIN

	3. Reading online data via framexmit::

		--data-source=framexmit --channel-name=H1=FAIKE-STRAIN

	4. Many other combinations possible, please add some!
	"""
	group = optparse.OptionGroup(parser, "Data source options", "Use these options to set up the appropriate data source")
	group.add_option("--data-source", metavar = "source", help = "Set the data source from [frames|framexmit|lvshm|nds|silence|white].  Required.")
	group.add_option("--block-size", type="int", metavar = "bytes", default = 16384 * 8 * 512, help = "Data block size to read in bytes. Default 16384 * 8 * 512 (512 seconds of double precision data at 16384 Hz.  This parameter is only used if --data-source is one of white, silence, AdvVirgo, LIGO, AdvLIGO, nds.")
	group.add_option("--frame-cache", metavar = "filename", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).")
	group.add_option("--frame-type", metavar = "name", action = "append", help = "Set the frame type for a given instrument.  Can be given multiple times as --frame-type=IFO=FRAME-TYPE. Used with --data-source=frames")
	group.add_option("--data-find-server", metavar = "url", help = "Set the data find server for LIGO data discovery. Used with --data-source=frames")
	group.add_option("--gps-start-time", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds. Required unless --data-source=lvshm")
	group.add_option("--gps-end-time", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds.  Required unless --data-source=lvshm")
	group.add_option("--injections", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load injections (optional).")
	group.add_option("--channel-name", metavar = "name", action = "append", help = "Set the name of the channels to process.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--nds-host", metavar = "hostname", help = "Set the remote host or IP address that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-port", metavar = "portnumber", type=int, default=31200, help = "Set the port of the remote host that serves nds data. This is required iff --data-source=nds")
	group.add_option("--nds-channel-type", metavar = "type", default = "online", help = "Set the port of the remote host that serves nds data. This is required only if --data-source=nds. default==online")
	group.add_option("--framexmit-addr", metavar = "name", action = "append", help = "Set the address of the framexmit service.  Can be given multiple times as --framexmit-addr=IFO=xxx.xxx.xxx.xxx:port")
	group.add_option("--framexmit-iface", metavar = "name", help = "Set the multicast interface address of the framexmit service.")
	group.add_option("--state-channel-name", metavar = "name", action = "append", help = "Set the name of the state vector channel.  This channel will be used to control the flow of data via the on/off bits.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--dq-channel-name", metavar = "name", action = "append", help = "Set the name of the data quality channel.  This channel will be used to control the flow of data via the on/off bits.  Can be given multiple times as --channel-name=IFO=CHANNEL-NAME")
	group.add_option("--shared-memory-partition", metavar = "name", action = "append", help = "Set the name of the shared memory partition for a given instrument.  Can be given multiple times as --shared-memory-partition=IFO=PARTITION-NAME")
	group.add_option("--shared-memory-assumed-duration", type = "int", default = 4, help = "Set the assumed span of files in seconds. Default = 4.")
	group.add_option("--shared-memory-block-size", type = "int", default = 4096, help = "Set the byte size to read per buffer. Default = 4096.")
	group.add_option("--frame-segments-file", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load frame segments.  Optional iff --data-source=frames")
	group.add_option("--frame-segments-name", metavar = "name", help = "Set the name of the segments to extract from the segment tables.  Required iff --frame-segments-file is given")
	group.add_option("--state-vector-on-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector on bits to process (optional).  The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	group.add_option("--state-vector-off-bits", metavar = "bits", default = [], action = "append", help = "Set the state vector off bits to process (optional).  The default is 0x160 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	group.add_option("--dq-vector-on-bits", metavar = "bits", default = [], action = "append", help = "Set the DQ vector on bits to process (optional).  The default is 0x7 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	group.add_option("--dq-vector-off-bits", metavar = "bits", default = [], action = "append", help = "Set the DQ vector off bits to process (optional).  The default is 0x160 for all detectors. Override with IFO=bits can be given multiple times.  Only currently has meaning for online (lvshm) data.")
	parser.add_option_group(group)


def mksegmentsrcgate(pipeline, src, segment_list, invert_output = False, rate = 1, **kwargs):
	"""
	Takes a segment list and produces a gate driven by it. Hook up your own input and output.

	@param kwargs passed through to pipeparts.mkgate(), e.g., used to set the gate's name.

	Gstreamer graph describing this function:

	.. graphviz::

	   digraph G {
	     compound=true;
	     node [shape=record fontsize=10 fontname="Verdana"];
	     rankdir=LR;
	     lal_segmentsrc;
	     lal_gate;
	     in [label="\<src\>"];
	     out [label="\<return value\>"];
	     in -> lal_gate -> out;
	     lal_segmentsrc -> lal_gate;
	   }

	"""
	return pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mkcapsfilter(pipeline, pipeparts.mksegmentsrc(pipeline, segment_list, invert_output = invert_output), caps = "audio/x-raw, rate=%d" % rate), **kwargs)


def mkbasicsrc(pipeline, gw_data_source_info, instrument, verbose = False):
	"""
	All the conditionals and stupid pet tricks for reading real or
	simulated h(t) data in one place.

	Consult the append_options() function and the GWDataSourceInfo class

	This src in general supports only one instrument although
	GWDataSourceInfo contains dictionaries of multi-instrument things.  By
	specifying the instrument when calling this function you will get ony a single
	instrument source.  A code wishing to have multiple basicsrcs will need to call
	this function for each instrument.

	**Gstreamer Graph**

	.. graphviz::

	   digraph mkbasicsrc {
		compound=true;
		node [shape=record fontsize=10 fontname="Verdana"];
		subgraph clusterfakesrc {
			fake_0 [label="fakesrc: white, silence, AdvVirgo, LIGO, AdvLIGO"];
			color=black;
			label="Possible path #1";
		}
		subgraph clusterframes {
			color=black;
			frames_0 [label="lalcachesrc: frames"];
			frames_1 [label ="framecppchanneldemux"];
			frames_2 [label ="queue"];
			frames_3 [label ="gate (if user provides segments)", style=filled, color=lightgrey];
			frames_4 [label ="audiorate"];
			frames_0 -> frames_1 -> frames_2 -> frames_3 ->frames_4;
			label="Possible path #2";
		}
		subgraph clusteronline {
			color=black;
			online_0 [label="lvshmsrc|framexmit"];
			online_1 [label ="framecppchanneldemux"];
			online_2a [label ="strain queue"];
			online_2b [label ="statevector queue"];
			online_3 [label ="statevector"];
			online_4 [label ="gate"];
			online_5 [label ="audiorate"];
			online_6 [label ="queue"];
			online_0 -> online_1;
			online_1 -> online_2a;
			online_1 -> online_2b;
			online_2b -> online_3;
			online_2a -> online_4;
			online_3 -> online_4 -> online_5 -> online_6;
			label="Possible path #3";
		}
		subgraph clusternds {
			nds_0 [label="ndssrc"];
			color=black;
			label="Possible path #4";
		}
		audioconv [label="audioconvert"];
		progress [label="progressreport (if verbose)", style=filled, color=lightgrey];
		sim [label="lalsimulation (if injections requested)", style=filled, color=lightgrey];
		queue [label="queue (if injections requested)", style=filled, color=lightgrey];

		// The connections
		fake_0 -> audioconv [ltail=clusterfakesrc];
		frames_4 -> audioconv [ltail=clusterframes];
		online_6 -> audioconv [ltail=clusteronline];
		nds_0 -> audioconv [ltail=clusternds];
		audioconv -> progress -> sim -> queue -> "?";
	   }

	"""
	statevector = dqvector = None

	# NOTE: timestamp_offset is a hack to allow seeking with fake
	# sources, a real solution should be fixing the general timestamp
	# problem which would allow seeking to work properly
	if gw_data_source_info.data_source == "white":
		src = pipeparts.mkfakesrc(pipeline, instrument, gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, volume = 1.0, timestamp_offset = int(gw_data_source_info.seg[0]) * Gst.SECOND)
	elif gw_data_source_info.data_source == "silence":
		src = pipeparts.mkfakesrc(pipeline, instrument, gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size, wave = 4, timestamp_offset = int(gw_data_source_info.seg[0]) * Gst.SECOND)
	elif gw_data_source_info.data_source == "LIGO":
		src = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "AdvLIGO":
		src = pipeparts.mkfakeadvLIGOsrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "AdvVirgo":
		src = pipeparts.mkfakeadvvirgosrc(pipeline, instrument = instrument, channel_name = gw_data_source_info.channel_dict[instrument], blocksize = gw_data_source_info.block_size)
	elif gw_data_source_info.data_source == "frames":
		if instrument == "V1":
			# FIXME Hack because virgo often just uses "V" in
			# the file names rather than "V1".  We need to
			# sieve on "V"
			src = pipeparts.mklalcachesrc(pipeline, location = gw_data_source_info.frame_cache, cache_src_regex = "V")
		else:
			src = pipeparts.mklalcachesrc(pipeline, location = gw_data_source_info.frame_cache, cache_src_regex = instrument[0], cache_dsc_regex = instrument)
		demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, channel_list = list(map("%s:%s".__mod__, gw_data_source_info.channel_dict.items())))
		pipeparts.framecpp_channeldemux_set_units(demux, dict.fromkeys(demux.get_property("channel-list"), "strain"))
		# allow frame reading and decoding to occur in a diffrent
		# thread
		src = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 8 * Gst.SECOND)
		pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), src.get_static_pad("sink"))
		if gw_data_source_info.frame_segments[instrument] is not None:
			# FIXME:  make segmentsrc generate segment samples
			# at the sample rate of h(t)?
			# FIXME:  make gate leaky when I'm certain that
			# will work.
			src = pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mksegmentsrc(pipeline, gw_data_source_info.frame_segments[instrument]), name = "%s_frame_segments_gate" % instrument)
			pipeparts.framecpp_channeldemux_check_segments.set_probe(src.get_static_pad("src"), gw_data_source_info.frame_segments[instrument])
		# FIXME:  remove this when pipeline can handle disconts
		src = pipeparts.mkaudiorate(pipeline, src, skip_to_first = True, silent = False)
	elif gw_data_source_info.data_source in ("framexmit", "lvshm"):
		# See https://wiki.ligo.org/DAC/ER2DataDistributionPlan#LIGO_Online_DQ_Channel_Specifica
		state_vector_on_bits, state_vector_off_bits = gw_data_source_info.state_vector_on_off_bits[instrument]
		dq_vector_on_bits, dq_vector_off_bits = gw_data_source_info.dq_vector_on_off_bits[instrument]

		if gw_data_source_info.data_source == "lvshm":
			# FIXME make wait_time adjustable through web
			# interface or command line or both
			src = pipeparts.mklvshmsrc(pipeline, shm_name = gw_data_source_info.shm_part_dict[instrument], assumed_duration = gw_data_source_info.shm_assumed_duration, blocksize = gw_data_source_info.shm_block_size, wait_time = 120)
		elif gw_data_source_info.data_source == "framexmit":
			src = pipeparts.mkframexmitsrc(pipeline, multicast_iface = gw_data_source_info.framexmit_iface, multicast_group = gw_data_source_info.framexmit_addr[instrument][0], port = gw_data_source_info.framexmit_addr[instrument][1], wait_time = 120)
		else:
			# impossible code path
			raise ValueError(gw_data_source_info.data_source)

		# 10 minutes of buffering, then demux
		src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * 60 * 10)
		src = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True)

		# extract state vector and DQ vector and convert to
		# booleans
		if gw_data_source_info.dq_channel_dict[instrument] == gw_data_source_info.state_channel_dict[instrument]:
			dqstatetee = pipeparts.mktee(pipeline, None)
			statevectorelem = statevector = pipeparts.mkstatevector(pipeline, dqstatetee, required_on = state_vector_on_bits, required_off = state_vector_off_bits, name = "%s_state_vector" % instrument)
			dqvectorelem = dqvector = pipeparts.mkstatevector(pipeline, dqstatetee, required_on = dq_vector_on_bits, required_off = dq_vector_off_bits, name = "%s_dq_vector" % instrument)
			pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.state_channel_dict[instrument]), dqstatetee.get_static_pad("sink"))
		else:
			# DQ and state vector are distinct channels
			# first DQ
			dqvectorelem = dqvector = pipeparts.mkstatevector(pipeline, None, required_on = dq_vector_on_bits, required_off = dq_vector_off_bits, name = "%s_dq_vector" % instrument)
			pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.dq_channel_dict[instrument]), dqvector.get_static_pad("sink"))
			# then State
			statevectorelem = statevector = pipeparts.mkstatevector(pipeline, None, required_on = state_vector_on_bits, required_off = state_vector_off_bits, name = "%s_state_vector" % instrument)
			pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.state_channel_dict[instrument]), statevector.get_static_pad("sink"))
		@bottle.route("/%s/statevector_on.txt" % instrument)
		def state_vector_state(elem = statevectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			on = elem.get_property("on-samples")
			return "%.9f %d" % (t, on)
		@bottle.route("/%s/statevector_off.txt" % instrument)
		def state_vector_state(elem = statevectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			off = elem.get_property("off-samples")
			return "%.9f %d" % (t, off)
		@bottle.route("/%s/statevector_gap.txt" % instrument)
		def state_vector_state(elem = statevectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			gap = elem.get_property("gap-samples")
			return "%.9f %d" % (t, gap)
		@bottle.route("/%s/dqvector_on.txt" % instrument)
		def dq_vector_state(elem = dqvectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			on = elem.get_property("on-samples")
			return "%.9f %d" % (t, on)
		@bottle.route("/%s/dqvector_off.txt" % instrument)
		def dq_vector_state(elem = dqvectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			off = elem.get_property("off-samples")
			return "%.9f %d" % (t, off)
		@bottle.route("/%s/dqvector_gap.txt" % instrument)
		def dq_vector_state(elem = dqvectorelem):
			t = float(lal.UTCToGPS(time.gmtime()))
			gap = elem.get_property("gap-samples")
			return "%.9f %d" % (t, gap)

		# extract strain with 1 buffer of buffering
		strain = pipeparts.mkqueue(pipeline, None, max_size_buffers = 1, max_size_bytes = 0, max_size_time = 0)
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), strain.get_static_pad("sink"))
		pipeparts.framecpp_channeldemux_set_units(src, {"%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]): "strain"})

		# fill in holes, skip duplicate data
		statevector = pipeparts.mkaudiorate(pipeline, statevector, skip_to_first = True, silent = False)
		dqvector = pipeparts.mkaudiorate(pipeline, dqvector, skip_to_first = True, silent = False)
		src = pipeparts.mkaudiorate(pipeline, strain, skip_to_first = True, silent = False, name = "%s_strain_audiorate" % instrument)
		@bottle.route("/%s/strain_dropped.txt" % instrument)
		# FIXME don't hard code the sample rate
		def strain_add(elem = src, rate = 16384):
			t = float(lal.UTCToGPS(time.gmtime()))
			# yes I realize we are reading the "add" property for a
			# route called dropped. That is because the data which
			# is "dropped" on route is "added" by the audiorate
			# element
			add = elem.get_property("add")
			return "%.9f %d" % (t, add // rate)

		# use state vector and DQ vector to gate strain.  the sizes
		# of the queues on the control inputs are not important.
		# they must be large enough to buffer the state vector
		# streams until they are needed, but the streams will be
		# consumed immediately when needed so there is no risk that
		# these queues actually fill up or add latency.  be
		# generous.
		statevector = pipeparts.mktee(pipeline, statevector)
		dqvector = pipeparts.mktee(pipeline, dqvector)
		src = pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mkqueue(pipeline, statevector, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), default_state = False, name = "%s_state_vector_gate" % instrument)
		src = pipeparts.mkgate(pipeline, src, threshold = 1, control = pipeparts.mkqueue(pipeline, dqvector, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), default_state = False, name = "%s_dq_vector_gate" % instrument)
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
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = Gst.SECOND * 64)

	#
	# done
	#

	return src, statevector, dqvector


def mkhtgate(pipeline, src, control = None, threshold = 8.0, attack_length = 128, hold_length = 128, **kwargs):
	"""
	A convenience function to provide thresholds on input data.  This can
	be used to remove large spikes / glitches etc.  Of course you can use it for
	other stuff by plugging whatever you want as input and ouput

	NOTE:  the queues constructed by this code assume the attack and
	hold lengths combined are less than 1 second in duration.

	**Gstreamer Graph**

	.. graphviz::

	   digraph G {
		compound=true;
		node [shape=record fontsize=10 fontname="Verdana"];
		rankdir=LR;
		tee ;
		inputqueue ;
		lal_gate ;
		in [label="\<src\>"];
		out [label="\<return\>"];
		in -> tee -> inputqueue -> lal_gate -> out;
		tee -> lal_gate;
	   }

	"""
	# FIXME someday explore a good bandpass filter
	# src = pipeparts.mkaudiochebband(pipeline, src, low_frequency, high_frequency)
	if control is None:
		control = src = pipeparts.mktee(pipeline, src)
	src = pipeparts.mkqueue(pipeline, src, max_size_time = Gst.SECOND, max_size_bytes = 0, max_size_buffers = 0)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = threshold, attack_length = -attack_length, hold_length = -hold_length, invert_control = True, **kwargs)

# Unit tests
if __name__ == "__main__":
	import doctest
	doctest.testmod()
