#!/usr/bin/env python3
# Copyright (C) 2018  Aaron Viets
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
import time
import resource

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
parser.add_option("--ifo", metavar = "name", help = "Name of interferometer (either L1 or H1)")
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time when to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time when to stop processing data")
parser.add_option("--frame-cache", metavar = "name", help = "Frame cache file with input data")
parser.add_option("--frame-type", metavar = "name", help = "Type of frame file, e.g., H1_R or L1_R for raw frames, H1_HOFT_C00 or L1_HOFT_C00 for low-latency production frames, etc.")
parser.add_option("--channel-prefix", metavar = "name", default = "", help = "Prefix for channels to read, also given to output channels (default is no prefix)")
parser.add_option("--channel-suffix", metavar = "name", default = "", help = "Suffix for channels to read, also given to output channels (default is no suffix)")
parser.add_option("--frame-length", metavar = "seconds", type = int, default = 64, help = "Length of output frames in seconds (Default = 64)")
parser.add_option("--frames-per-file", type = int, default = 1, help = "Number of frames per frame file (Default = 1)")
parser.add_option("--output-path", metavar = "name", help = "Location to write output frames to")

options, filenames = parser.parse_args()

#
# 
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


#
# This pipeline reads the channels needed for calibration from the raw frames
# and writes them to smaller frames for faster access. It can also change the
# frame length if desired
#

ifo = options.ifo
frame_cache = options.frame_cache
frame_type = options.frame_type
output_path = options.output_path
frame_length = options.frame_length
frames_per_file = options.frames_per_file
chan_prefix = options.channel_prefix
chan_suffix = options.channel_suffix
channel_list = []

# Get a list of available channels from frchannels.txt, a file which should be written before running this script
# FIXME: I'm sure there's a better way to do this. It would be nice to check the first frame file in the cache using Python alone.
try:
	f = open('frchannels.txt', 'r')
	available_channels = f.read()
	f.close()
	print("Making channel list using available channels")
except:
	print("Cannot find file frchannels.txt. Run FrChannels to check for available channels")

# These are (or may be) in the calibrated frames
channel_list.append(chan_prefix + "CALIB_STRAIN" + chan_suffix)
channel_list.append("ODC-MASTER_CHANNEL_OUT_DQ")
channel_list.append(chan_prefix + "CALIB_STATE_VECTOR" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_STRAIN_CLEAN" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_TST_REAL" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_TST_IMAGINARY" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_TST_REAL_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_TST_IMAGINARY_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_A_REAL" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_A_IMAGINARY" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PU_REAL" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PU_IMAGINARY" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PU_REAL_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PU_IMAGINARY_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PUM_REAL" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PUM_IMAGINARY" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PUM_REAL_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_PUM_IMAGINARY_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_UIM_REAL" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_UIM_IMAGINARY" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_UIM_REAL_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_UIM_IMAGINARY_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_C" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_KAPPA_C_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_F_CC" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_F_CC_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_F_S" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_SRC_Q_INVERSE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_F_S_NOGATE" + chan_suffix)
channel_list.append(chan_prefix + "CALIB_SRC_Q_INVERSE_NOGATE" + chan_suffix)

# These are (or may be) in the raw frames
channel_list.append("CAL-INJ_STATUS_OUT_DQ")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCALY_LINE1_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCAL_LINE3_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_UNCERTAINTY")
channel_list.append("CAL-CS_LINE_SUM_DQ")
channel_list.append("SUS-ETMY_L1_CAL_LINE_OUT_DQ")
channel_list.append("SUS-ETMY_L2_CAL_LINE_OUT_DQ")
channel_list.append("SUS-ETMY_L3_CAL_LINE_OUT_DQ")
channel_list.append("SUS-ETMX_L1_CAL_LINE_OUT_DQ")
channel_list.append("SUS-ETMX_L2_CAL_LINE_OUT_DQ")
channel_list.append("SUS-ETMX_L3_CAL_LINE_OUT_DQ")
channel_list.append("CAL-PCALY_RX_PD_OUT_DQ")
channel_list.append("CAL-PCALY_TX_PD_OUT_DQ")
channel_list.append("CAL-PCALX_RX_PD_OUT_DQ")
channel_list.append("CAL-PCALX_TX_PD_OUT_DQ")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALY_LINE1_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALX_LINE1_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALX_LINE2_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE3_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALY_LINE3_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALX_LINE3_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_COMPARISON_OSC_FREQ")
channel_list.append("CAL-CS_TDEP_PCALX_LINE4_COMPARISON_OSC_FREQ")
channel_list.append("CAL-PCALX_PCALOSC1_OSC_FREQ")
channel_list.append("PEM-EY_MAINSMON_EBAY_1_DQ")
channel_list.append("PEM-EY_MAINSMON_EBAY_2_DQ")
channel_list.append("PEM-EY_MAINSMON_EBAY_3_DQ")
channel_list.append("PEM-EY_MAINSMON_EBAY_QUAD_SUM_DQ")
channel_list.append("PEM-EX_MAINSMON_EBAY_1_DQ")
channel_list.append("PEM-EX_MAINSMON_EBAY_2_DQ")
channel_list.append("PEM-EX_MAINSMON_EBAY_3_DQ")
channel_list.append("PEM-EX_MAINSMON_EBAY_QUAD_SUM_DQ")
channel_list.append("PEM-CS_MAINSMON_EBAY_1_DQ")
channel_list.append("PEM-CS_MAINSMON_EBAY_2_DQ")
channel_list.append("PEM-CS_MAINSMON_EBAY_3_DQ")
channel_list.append("PEM-CS_MAINSMON_EBAY_QUAD_SUM_DQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_CORRECTION_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_CORRECTION_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_CORRECTION_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_CORRECTION_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE3_CORRECTION_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE3_CORRECTION_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_CORRECTION_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_CORRECTION_IMAG")
channel_list.append("CAL-CS_TDEP_REF_INVA_CLGRATIO_TST_REAL")
channel_list.append("CAL-CS_TDEP_REF_INVA_CLGRATIO_TST_IMAG")
channel_list.append("CAL-CS_TDEP_REF_CLGRATIO_CTRL_REAL")
channel_list.append("CAL-CS_TDEP_REF_CLGRATIO_CTRL_IMAG")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_USUM_INV_REAL")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_USUM_INV_IMAG")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_USUM_REAL")
channel_list.append("CAL-CS_TDEP_DARM_LINE1_REF_A_USUM_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_C_NOCAVPOLE_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_C_NOCAVPOLE_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_D_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_D_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_A_USUM_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE2_REF_A_USUM_IMAG")
channel_list.append("CAL-CS_TDEP_ESD_LINE1_REF_A_TST_NOLOCK_REAL")
channel_list.append("CAL-CS_TDEP_ESD_LINE1_REF_A_TST_NOLOCK_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_C_NOCAVPOLE_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_C_NOCAVPOLE_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_D_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_D_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_A_USUM_REAL")
channel_list.append("CAL-CS_TDEP_PCALY_LINE4_REF_A_USUM_IMAG")
channel_list.append("CAL-DARM_CTRL_WHITEN_OUT_DBL_DQ")
channel_list.append("CAL-DARM_ERR_WHITEN_OUT_DBL_DQ")
channel_list.append("CAL-DELTAL_EXTERNAL_DQ")
channel_list.append("CAL-CFTD_DELTAL_EXTERNAL_DQ")
channel_list.append("CAL-DELTAL_CTRL_TST_DBL_DQ")
channel_list.append("CAL-DELTAL_CTRL_PUM_DBL_DQ")
channel_list.append("CAL-DELTAL_CTRL_UIM_DBL_DQ")
channel_list.append("CAL-DELTAL_RESIDUAL_DBL_DQ")
channel_list.append("PSL-DIAG_BULLSEYE_WID_OUT_DQ")
channel_list.append("PSL-DIAG_BULLSEYE_PIT_OUT_DQ")
channel_list.append("PSL-DIAG_BULLSEYE_YAW_OUT_DQ")
channel_list.append("PSL-ISS_SECONDLOOP_OUTPUT_DQ")
channel_list.append("IMC-WFS_A_DC_PIT_OUT_DQ")
channel_list.append("IMC-WFS_B_DC_PIT_OUT_DQ")
channel_list.append("IMC-WFS_A_DC_YAW_OUT_DQ")
channel_list.append("IMC-WFS_B_DC_YAW_OUT_DQ")
channel_list.append("LSC-SRCL_IN1_DQ")
channel_list.append("LSC-MICH_IN1_DQ")
channel_list.append("LSC-PRCL_IN1_DQ")
channel_list.append("ASC-DHARD_P_OUT_DQ")
channel_list.append("ASC-DHARD_Y_OUT_DQ")
channel_list.append("ASC-CHARD_P_OUT_DQ")
channel_list.append("ASC-CHARD_Y_OUT_DQ")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_PUM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_PUM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_UIM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_A_UIM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_C_NOCAVPOLE_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_C_NOCAVPOLE_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_D_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_REF_D_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE2_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_REF_A_UIM_NOLOCK_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_REF_A_UIM_NOLOCK_REAL")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_REF_INVA_UIM_RESPRATIO_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE1_REF_INVA_UIM_RESPRATIO_REAL")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_REF_A_PUM_NOLOCK_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_REF_A_PUM_NOLOCK_REAL")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_REF_INVA_PUM_RESPRATIO_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_REF_INVA_PUM_RESPRATIO_REAL")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_REF_A_TST_NOLOCK_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_REF_A_TST_NOLOCK_REAL")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_REF_INVA_TST_RESPRATIO_IMAG")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_REF_INVA_TST_RESPRATIO_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_PUM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_PUM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_UIM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_A_UIM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_C_NOCAVPOLE_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_C_NOCAVPOLE_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_D_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_REF_D_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE4_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_PUM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_PUM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_TST_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_TST_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_UIM_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_A_UIM_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_C_NOCAVPOLE_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_C_NOCAVPOLE_REAL")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_D_IMAG")
channel_list.append("CAL-CS_TDEP_PCAL_LINE1_REF_D_REAL")
channel_list.append("CAL-DARM_CTRL_DBL_DQ")
channel_list.append("CAL-DARM_ERR_DBL_DQ")
channel_list.append("CAL-CS_TDEP_SUS_LINE2_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_SUS_LINE3_UNCERTAINTY")
channel_list.append("CAL-CS_TDEP_KAPPA_UIM_REAL_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_UIM_IMAG_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_PUM_REAL_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_PUM_IMAG_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_TST_REAL_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_TST_IMAG_OUTPUT")
channel_list.append("CAL-CS_TDEP_KAPPA_C_OUTPUT")
channel_list.append("CAL-CS_TDEP_F_C_OUTPUT")
channel_list.append("CAL-CS_TDEP_F_S_OUTPUT")
channel_list.append("CAL-CS_TDEP_Q_S_OUTPUT")
channel_list.append("GRD-IFO_OK")
#channel_list.append("ISC_LOCK_STATE_N")
#channel_list.append("ISC_LOCK_STATUS")
channel_list.append("GRD-ISC_LOCK_OK")
channel_list.append("GRD-ISC_LOCK_ERROR")
channel_list.append("GRD-IFO_INTENT")
channel_list.append("GRD-IFO_READY")
temp_list = channel_list
channel_list = []
for chan in temp_list:
	if chan in available_channels:
		channel_list.append(chan)

print("Finished channel list")

ifo_channel_list = []
for chan in channel_list:
	ifo_channel_list.append((ifo, chan))

def frame_manipulator(pipeline, name):

	#
	# This pipeline reads the channels needed for calibration from the raw frames
	# and writes them to smaller frames for faster access.
	#

	head_dict = {}

	# Get the data from the raw frames and pick out the channels we want
	src = pipeparts.mklalcachesrc(pipeline, location = frame_cache, cache_dsc_regex = ifo)
	src = pipeparts.mkprogressreport(pipeline, src, "start")
	demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, ifo_channel_list)))

	# Make a muxer to collect the channels we need
	channelmux_input_dict = {}
	for key, chan in zip(channel_list, channel_list):
		head_dict[key] = calibration_parts.hook_up(pipeline, demux, chan, ifo, 1.0)
		head_dict[key] = pipeparts.mkprogressreport(pipeline, head_dict[key], "before muxer %s" % key)
		channelmux_input_dict["%s:%s" % (ifo, chan)] = calibration_parts.mkqueue(pipeline, head_dict[key])

	mux = pipeparts.mkframecppchannelmux(pipeline, channelmux_input_dict, frame_duration = frame_length, frames_per_file = frames_per_file, compression_scheme = 6, compression_level = 3)

	mux = pipeparts.mkprogressreport(pipeline, mux, "end")
	pipeparts.mkframecppfilesink(pipeline, mux, frame_type = frame_type, path = output_path, instrument = ifo)

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


test_common.build_and_run(frame_manipulator, "frame_manipulator", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))


