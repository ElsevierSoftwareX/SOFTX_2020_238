#!/usr/bin/env python3
# Copyright (C) 2020  Aaron Viets
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
#				   Preamble
#
# =============================================================================
#


import sys
import os
import numpy
from math import pi

from optparse import OptionParser, Option

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import simplehandler
from gstlal import datasource

from ligo import segments
from gstlal import test_common

parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--frame-cache", metavar = "name", type = str, help = "Name of frame cache file that contains the data")
parser.add_option("--channel-list", metavar = "list", type = str, default = 'DCS-CALIB_KAPPA_TST_IMAGINARY_C01,DCS-CALIB_KAPPA_PUM_IMAGINARY_C01,DCS-CALIB_KAPPA_UIM_IMAGINARY_C01', help = "Comma-separated list of channel-names to read in from the frames")
parser.add_option("--statevector-channel", metavar = "name", type = str, default = None, help = "Name of state vector channel used to gate the data.  If not set, data is not gated.")
parser.add_option("--statevector-bitmask", type = int, default = 1, help = "Bitmask used to determine which bits of the state vector must be on for the data to not be gated.")
parser.add_option("--sample-rate", metavar = "Hz", type = int, default = 16, help = "Sample rate of the channels")
parser.add_option("--average-time", metavar = "seconds", type = float, default = 14400, help = "Duration (in seconds) of data to average before checking for changes")
parser.add_option("--detection-threshold", type = float, default = 0.005, help = "Minimum magnitude of difference needed for detection.  Channels are summed before checking for changes.")
parser.add_option("--filename", metavar = "name", type = str, default = None, help = "Name of file in which to write output.  If not given, no file is written")

options, filenames = parser.parse_args()


# Get channel list
channel_list = options.channel_list.split(',')
ifo_channel_list = []
for chan in channel_list:
	ifo_channel_list.append((options.ifo, chan))
if options.statevector_channel is not None:
	ifo_channel_list.append((options.ifo, options.statevector_channel))

# 
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


#
# This pipeline sums the given channels and checks for significant changes.
#


def detect_change(pipeline, name):
	# Get the data from the frames
	data = pipeparts.mklalcachesrc(pipeline, location = options.frame_cache, cache_dsc_regex = options.ifo)
	data = pipeparts.mkframecppchanneldemux(pipeline, data, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, ifo_channel_list)))

	streams = []
	for chan in channel_list:
		stream = calibration_parts.hook_up(pipeline, data, chan, options.ifo, 64.0)
		streams.append(stream)
	summed_streams = calibration_parts.mkadder(pipeline, streams)

	if options.statevector_channel is not None:
		state_vector = calibration_parts.hook_up(pipeline, data, options.statevector_channel, options.ifo, 64.0)
		state_vector = pipeparts.mkgeneric(pipeline, state_vector, "lal_logicalundersample", required_on = 1)
		summed_streams = calibration_parts.mkgate(pipeline, summed_streams, state_vector, threshold = 1.0)

	summed_streams = pipeparts.mkgeneric(pipeline, summed_streams, "lal_detectchange", average_samples = int(options.average_time * options.sample_rate), detection_threshold = options.detection_threshold, filename = options.filename)
	pipeparts.mkfakesink(pipeline, summed_streams)

	#
	# done
	#

	return pipeline

#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


# Run pipeline
test_common.build_and_run(detect_change, "detect_change", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))


