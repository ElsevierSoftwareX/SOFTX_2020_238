#!/usr/bin/env python
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
import ConfigParser

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import test_common
from gstlal import simplehandler
from gstlal import datasource

from glue.ligolw import ligolw
from glue.ligolw import array
from glue.ligolw import param
from glue.ligolw.utils import segments as ligolw_segments
array.use_in(ligolw.LIGOLWContentHandler)
param.use_in(ligolw.LIGOLWContentHandler)
from glue.ligolw import utils
from ligo import segments

parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--raw-frame-cache", metavar = "name", help = "Raw frame cache file")
parser.add_option("--calibrated-frame-cache", metavar = "name", help = "Calibrated frame cache file")
parser.add_option("--config-file", metavar = "name", help = "Configurations file used to produce GDS/DCS calibrated frames, needed to get pcal line frequencies and correction factors")
parser.add_option("--pcal-channel-name", metavar = "name", default = "CAL-PCALY_TX_PD_OUT_DQ", help = "Name of the pcal channel you wish to use")
parser.add_option("--calibrated-channel-list", metavar = "list", default = None, help = "Comma-separated list of calibrated channels to compare to pcal")
parser.add_option("--calcs-channel-list", metavar = "list", default = None, help = "Comma-separated list of calibrated channels in the raw frames to compare to pcal")

options, filenames = parser.parse_args()

# Read the config file
def ConfigSectionMap(section):
	dict1 = {}
	options = Config.options(section)
	for option in options:
		try:
			dict1[option] = Config.get(section, option)
			if dict1[option] == -1:
				DebugPrint("skip: %s" % option)
		except:
			print("exception on %s!" % option)
			dict[option] = None
	return dict1

Config = ConfigParser.ConfigParser()
Config.read(options.config_file)

InputConfigs = ConfigSectionMap("InputConfigurations")

#
# Load in the filters file that contains filter coefficients, etc.
#

# Search the directory tree for files with names matching the one we want.
filters_name = InputConfigs["filtersfilename"]
filters_paths = []
# Check the user's home directory
for dirpath, dirs, files in os.walk(os.environ['HOME']):
	if filters_name in files:
		# We prefer filters that came directly from a GDSFilters directory of the calibration SVN
		if dirpath.count("GDSFilters") > 0:
			filters_paths.insert(0, os.path.join(dirpath, filters_name))
		else:
			filters_paths.append(os.path.join(dirpath, filters_name))
if not len(filters_paths):
	raise ValueError("Cannot find filters file %s in home directory %s or in /ligo/svncommon/CalSVN/aligocalibration/trunk/Runs/*/GDSFilters", (filters_name, os.environ['HOME']))
print "\nLoading calibration filters from %s\n" % filters_paths[0]
filters = numpy.load(filters_paths[0])

ifo = options.ifo

# Set up channel lists
channel_list = []
calibrated_channel_list = []
calcs_channel_list = []
channel_list.append((ifo, options.pcal_channel_name))
if options.calibrated_channel_list is not None:
	calibrated_channels = options.calibrated_channel_list.split(',')
	for channel in calibrated_channels:
		channel_list.append((ifo, channel))
else:
	calibrated_channels = []
if options.calcs_channel_list is not None:
	calcs_channels = options.calcs_channel_list.split(',')
	for channel in calcs_channels:
		channel_list.append((ifo, channel))
else:
	calcs_channels = []

# Read stuff we need from the filters file
frequencies = [float(filters['ka_pcal_line_freq']), float(filters['kc_pcal_line_freq']), float(filters['high_pcal_line_freq'])]
pcal_corrections= [float(filters['ka_pcal_corr_re']), float(filters['ka_pcal_corr_im']), float(filters['kc_pcal_corr_re']), float(filters['kc_pcal_corr_im']), float(filters['high_pcal_corr_re']), float(filters['high_pcal_corr_im'])]
demodulated_pcal_list = []
try:
	arm_length = float(filters['arm_length'])
except:
	arm_length = 3995.1

# demodulation and averaging parameters
filter_time = 20
average_time = 1
rate_out = 1

#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


def pcal2darm(pipeline, name):

	# Get pcal from the raw frames
	raw_data = pipeparts.mklalcachesrc(pipeline, location = options.raw_frame_cache, cache_dsc_regex = ifo)
	raw_data = pipeparts.mkframecppchanneldemux(pipeline, raw_data, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	pcal = calibration_parts.hook_up(pipeline, raw_data, options.pcal_channel_name, ifo, 1.0)
	pcal = calibration_parts.caps_and_progress(pipeline, pcal, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", "pcal")
	pcal = pipeparts.mktee(pipeline, pcal)

	# Demodulate the pcal channel at the lines of interest
	for i in range(0, len(frequencies)):
		demodulated_pcal = calibration_parts.demodulate(pipeline, pcal, frequencies[i], True, rate_out, filter_time, 0.5, prefactor_real = pcal_corrections[2 * i], prefactor_imag = pcal_corrections[2 * i + 1])
		demodulated_pcal_list.append(pipeparts.mktee(pipeline, demodulated_pcal))

	# Check if we are taking pcal-to-darm ratios for CALCS data
	for channel in calcs_channels:
		calcs_deltal = calibration_parts.hook_up(pipeline, raw_data, channel, ifo, 1.0)
		calcs_deltal = calibration_parts.caps_and_progress(pipeline, calcs_deltal, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", channel)
		calcs_deltal = pipeparts.mktee(pipeline, calcs_deltal)
		for i in range(0, len(frequencies)):
			# Demodulate DELTAL_EXTERNAL at each line
			demodulated_calcs_deltal = calibration_parts.demodulate(pipeline, calcs_deltal, frequencies[i], True, rate_out, filter_time, 0.5)
			# Take ratio \DeltaL(f) / pcal(f)
			deltaL_over_pcal = calibration_parts.complex_division(pipeline, demodulated_calcs_deltaL, demodulated_pcal_list[i])
			# Take a running average
			deltaL_over_pcal = pipeparts.mkgeneric(pipeline, deltaL_over_pcal, "lal_smoothkappas", array_size = 1, avg_array_size = int(rate_out * average_time))
			# Write to file
			pipeparts.mknxydumpsink(pipeline, deltaL_over_pcal, "%s_%s_over_%s_at_line%d.txt" % (ifo, channel, options.pcal_channel_name, i + 1))

	# Check if we are taking pcal-to-darm ratios for gstlal calibrated data
	if options.calibrated_channel_list is not None:
		# Get calibrated channels from the calibrated frames
		hoft_data = pipeparts.mklalcachesrc(pipeline, location = options.calibrated_frame_cache, cache_dsc_regex = ifo)
		hoft_data = pipeparts.mkframecppchanneldemux(pipeline, hoft_data, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	for channel in calibrated_channels:
		hoft = calibration_parts.hook_up(pipeline, hoft_data, channel, ifo, 1.0)
		hoft = calibration_parts.caps_and_progress(pipeline, hoft, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", channel)
		deltal = pipeparts.mkaudioamplify(pipeline, hoft, arm_length)
		deltal = pipeparts.mktee(pipeline, deltal)
		for i in range(0, len(frequencies)):
			# Demodulate \DeltaL at each line
			demodulated_deltal = calibration_parts.demodulate(pipeline, deltal, frequencies[i], True, rate_out, filter_time, 0.5)
			# Take ratio \DeltaL(f) / pcal(f)
			deltaL_over_pcal = calibration_parts.complex_division(pipeline, demodulated_deltal, demodulated_pcal_list[i])
			# Take a running average
			deltaL_over_pcal = pipeparts.mkgeneric(pipeline, deltaL_over_pcal, "lal_smoothkappas", array_size = 1, avg_array_size = int(rate_out * average_time))
			# Write to file
			pipeparts.mknxydumpsink(pipeline, deltaL_over_pcal, "%s_%s_over_%s_at_line%d.txt" % (ifo, channel, options.pcal_channel_name, i + 1))

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


test_common.build_and_run(pcal2darm, "pcal2darm", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))


