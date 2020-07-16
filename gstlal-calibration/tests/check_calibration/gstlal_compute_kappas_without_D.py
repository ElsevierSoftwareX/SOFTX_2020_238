#!/usr/bin/env python3
# Copyright (C) 2019 Aaron Viets, Jack Mango
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


# This first section "imports" python packages (other codes) that we will need
# to use.

import numpy	# numpy is a library for python, including arrays and matrices,
		# as well as mathematical functions.

import sys 	# another Python thing

import os
import time
from math import pi
import resource
import datetime
import time

from gstlal import pipeparts	
# pipeparts is from gstlal. It contains python functions to build GStreamer
# pipelines. You can see the code here:
# /home/jack.mango/src/gstlal/gstlal/python/pipeparts/__init__.py

from gstlal import calibration_parts
# calibration_parts includes more pipeline-building tools:
# /home/jack.mango/src/gstlal/gstlal-calibration/python/calibration_parts.py

from gstlal import test_common # More pipeline-building and testing tools in this directory

from gi.repository import Gst # gstreamer stuff

from optparse import OptionParser, Option # Allows us to take in command line options

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from gstlal import simplehandler
from gstlal import datasource

from ligo import segments

# Parsing command line options
parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--frame-cache", metavar = "name", type = str, help = "Name of frame cache file that contains the raw data")
parser.add_option("--f1", metavar = "Hz", type = float, default = 17.1, help = "The frequency (in Hz) of the first Pcal line.  For O3, it's 17.1 Hz at H1 and 16.3 Hz at L1.  Default is H1's frequency, 17.1 Hz")
parser.add_option("--f2", metavar = "Hz", type = float, default = 410.3, help = "The frequency (in Hz) of the second Pcal line.  For O3, it's 410.3 Hz at H1 and 434.9 Hz at L1.  Default is H1's frequency, 410.3 Hz")
parser.add_option("--fT", metavar = "Hz", type = float, default = 17.6, help = "The frequency (in Hz) of the TST/L3/ESD line.  For O3, it's 17.6 Hz at H1 and 16.9 Hz at L1.  Default is H1's frequency, 17.6 Hz")
parser.add_option("--fP", metavar = "Hz", type = float, default = 16.4, help = "The frequency (in Hz) of the PUM/L2 line.  For O3, it's 16.4 Hz at H1 and 15.7 Hz at L1.  Default is H1's frequency, 16.4 Hz")
parser.add_option("--fU", metavar = "Hz", type = float, default = 15.6, help = "The frequency (in Hz) of the UIM/L1 line.  For O3, it's 15.6 Hz at H1 and 15.1 Hz at L1.  Default is H1's frequency, 15.6 Hz")

options, filenames = parser.parse_args()

# shortcut names for options we use frequently
ifo = options.ifo
frame_cache = options.frame_cache
f1 = options.f1
f2 = options.f2
fT = options.fT
fP = options.fP
fU = options.fU


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


# This pipeline reads in raw LIGO data.  As of now, it just demodulates the
# error signal and writes the result to a file.  For reference, here are the
# current frequencies of the calibration lines at both H1 and L1:
#______________________________________________________________________________
# Name	|   H1 frequency (Hz)	|   L1 frequency (Hz)	| Injected using
# f_1	|	17.1		|	16.3		|	Pcal
# f_2	|	410.3		|	434.9		|	Pcal
# f_T	|	17.6		|	16.9		|	TST actuator
# f_P	|	16.4		|	15.7		|	PUM actuator
# f_U	|	15.6		|	15.1		|	UIM actuator

# the channel list below helps the demuxer sort through the channels faster
channel_list = [(ifo, "CAL-DARM_ERR_DBL_DQ"), (ifo, "CAL-DARM_CTRL_DBL_DQ"), (ifo, "CAL-PCALY_RX_PD_OUT_DQ"), (ifo, "SUS-ETMX_L3_CAL_LINE_OUT_DQ"), (ifo, "SUS-ETMX_L2_CAL_LINE_OUT_DQ"), (ifo, "SUS-ETMX_L1_CAL_LINE_OUT_DQ")]

def gstlal_compute_kappas_without_D(pipeline, name):

	# The first line makes the element lal_cachesrc, which reads in raw data using the cache file
	source = pipeparts.mklalcachesrc(pipeline, location = frame_cache, cache_dsc_regex = ifo)

	# Next, the demuxer splits it into separate channels
	demux = pipeparts.mkframecppchanneldemux(pipeline, source, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))

	# Next, we use a function from calibration_parts.py that hooks up to the demuxer 
	# and does a few sanity checks on the data.  This uses multiple elements.
	darm_err = calibration_parts.hook_up(pipeline, demux, "CAL-DARM_ERR_DBL_DQ", ifo, 1.0)

	# Another calibration_parts function to set the "caps", which are stream parameters
	# that tell gstreamer the data type, sample rate, etc.  A progress report will produce
	# output to the screen so that you can see it is running.
	darm_err = calibration_parts.caps_and_progress(pipeline, darm_err, "audio/x-raw, format=F64LE, rate=%d, channels=1, channel-mask=(bitmask)0x0", "darm_err")

	# You will need to use the error signal in more than one place (you'll have to
	# demodulate it at all five frequencies), so we need a tee here.
	darm_err = pipeparts.mktee(pipeline, darm_err)

	# Finally, an interesting operation - now you need to demodulate the error signal.
	# You will have to do this to every input signal.  darm_err is currently a time series,
	# but you want to know its amplitude and phase at a single frequency.  To do this, we
	# Demodulate it at the specified frequency
	darm_err_at_f1 = calibration_parts.demodulate(pipeline, darm_err, f1, True, 16, 20, 0)

	# You'll also need this in multiple places, so here's a tee
	darm_err_at_f1 = pipeparts.mktee(pipeline, darm_err_at_f1)

	# This writes the data to a file, currently called "darm_err_at_f1.txt."  Eventually,
	# you will want to write the TDCFs to a file.
	pipeparts.mknxydumpsink(pipeline, darm_err_at_f1, "darm_err_at_f1.txt")

	# To move forward, you can remove the nxydumpsink above and use darm_err_at_f1 for
	# your calculations.  You will also need to demodulate the Pcal channel at f1.  And
	# you will have to do several more demodulations (10 total I think, since each of
	# the five lines is present in darm_err and in one injection channel).  Then, you
	# will need to do additions, multiplications, divisions, powers, etc.  I would
	# suggest looking in gstlal_compute_strain (the calibration pipeline) and
	# calibration_parts.py for examples of how to do these.  calibration_parts has lots
	# of functions that should be helpful.

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


# This function calls the above pipeline function to build and run the pipeline.
test_common.build_and_run(gstlal_compute_kappas_without_D, "gstlal_compute_kappas_without_D", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))


