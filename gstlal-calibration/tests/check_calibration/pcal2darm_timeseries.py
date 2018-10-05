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
from math import pi
import resource
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

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
from gstlal import test_common

parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--raw-frame-cache", metavar = "name", help = "Raw frame cache file")
parser.add_option("--calibrated-frame-cache", metavar = "name", help = "Calibrated frame cache file")
parser.add_option("--config-file", metavar = "name", help = "Configurations file used to produce GDS/DCS calibrated frames, needed to get pcal line frequencies and correction factors")
parser.add_option("--pcal-channel-name", metavar = "name", default = "CAL-PCALY_TX_PD_OUT_DQ", help = "Name of the pcal channel you wish to use")
parser.add_option("--calibrated-channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of calibrated channels to compare to pcal")
parser.add_option("--calcs-channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of calibrated channels in the raw frames to compare to pcal")
parser.add_option("--magnitude-ranges", metavar = "list", type = str, default = "0.97,1.03;0.95,1.05;0.8,1.2", help = "Ranges for magnitude plots. Semicolons separate ranges for different plots, and commas separate min and max values.")
parser.add_option("--phase-ranges", metavar = "list", type = str, default = "-1.0,1.0;-3.0,3.0;-10.0,10.0", help = "Ranges for phase plots, in degrees. Semicolons separate ranges for different plots, and commas separate min and max values.")

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
print "\nSearching for %s ..." % filters_name
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
print "Loading calibration filters from %s\n" % filters_paths[0]
filters = numpy.load(filters_paths[0])

ifo = options.ifo

# Set up channel lists
channel_list = []
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

if not len(options.magnitude_ranges.split(';')) == len(frequencies):
	raise ValueError("Number of magnitude ranges given is not equal to number of pcal line frequencies (%d != %d)." % (len(options.magnitude_ranges.split(';')), len(frequencies)))
if not len(options.phase_ranges.split(';')) == len(frequencies):
	raise ValueError("Number of phase ranges given is not equal to number of pcal line frequencies (%d != %d)." % (len(options.phase_ranges.split(';')), len(frequencies)))


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
			# Find the magnitude
			deltaL_over_pcal = pipeparts.mktee(pipeline, deltaL_over_pcal)
			magnitude = pipeparts.mkgeneric(pipeline, deltaL_over_pcal, "cabs")
			# Find the phase
			phase = pipeparts.mkgeneric(pipeline, deltaL_over_pcal, "carg")
			phase = pipeparts.mkaudioamplify(pipeline, phase, 180.0 / numpy.pi)
			# Interleave
			magnitude_and_phase = calibration_parts.mkinterleave(pipeline, [magnitude, phase])
			# Write to file
			pipeparts.mknxydumpsink(pipeline, magnitude_and_phase, "%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, channel, options.pcal_channel_name, frequencies[i], options.gps_start_time))

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

# Read data from files and plot it
colors = ['r.', 'g.', 'y.', 'c.', 'm.', 'b.'] # Hopefully the user will not want to plot more than six datasets on one plot.
channels = calcs_channels
channels.extend(calibrated_channels)
for i in range(0, len(frequencies)):
	data = numpy.loadtxt("%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, channels[0], options.pcal_channel_name, frequencies[i], options.gps_start_time))
	t_start = data[0][0]
	dur = data[len(data) - 1][0] - t_start
	t_unit = 'seconds'
	sec_per_t_unit = 1.0
	if dur > 60 * 60 * 100:
		t_unit = 'days'
		sec_per_t_unit = 60.0 * 60.0 * 24.0
	elif dur > 60 * 100:
		t_unit = 'hours'
		sec_per_t_unit = 60.0 * 60.0
	elif dur > 100:
		t_unit = 'minutes'
		sec_per_t_unit = 60.0
	times = []
	magnitudes = [[]]
	phases = [[]]
	for k in range(0, len(data)):
		times.append((data[k][0] - t_start) / sec_per_t_unit)
		magnitudes[0].append(data[k][1])
		phases[0].append(data[k][2])
	markersize = 5000.0 / dur
	markersize = min(markersize, 2.0)
	markersize = max(markersize, 0.2)
	# Make plots
	plt.figure(figsize = (10, 10))
	plt.subplot(211)
	plt.plot(times, magnitudes[0], colors[0], markersize = markersize, label = '%s [avg = %0.5f, std = %0.5f]' % (channels[0], numpy.mean(magnitudes[0]), numpy.std(magnitudes[0])))
	plt.title(r'%s Delta $L_{\rm free}$ / Pcal at %0.1f Hz' % ( ifo, frequencies[i]))
	plt.ylabel('Magnitude')
	magnitude_range = options.magnitude_ranges.split(';')[i]
	plt.ylim(float(magnitude_range.split(',')[0]), float(magnitude_range.split(',')[1]))
	plt.grid(True)
	leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.5)
	plt.subplot(212)
	plt.plot(times, phases[0], colors[0], markersize = markersize, label = '%s [avg = %0.5f, std = %0.5f]' % (channels[0], numpy.mean(phases[0]), numpy.std(phases[0])))
	leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.5)
	plt.ylabel('Phase [deg]')
	plt.xlabel('Time in %s since %s UTC' % (t_unit, time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t_start + 315964782))))
	phase_range = options.phase_ranges.split(';')[i]
	plt.ylim(float(phase_range.split(',')[0]), float(phase_range.split(',')[1]))
	plt.grid(True)
	for j in range(1, len(channels)):
		data = numpy.loadtxt("%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, channels[j], options.pcal_channel_name, frequencies[i], options.gps_start_time))
		magnitudes.append([])
		phases.append([])
		for k in range(0, len(data)):
			magnitudes[j].append(data[k][1])
			phases[j].append(data[k][2])
		plt.subplot(211)
		plt.plot(times, magnitudes[j], colors[j], markersize = markersize, label = '%s [avg = %0.5f, std = %0.5f]' % (channels[j], numpy.mean(magnitudes[j]), numpy.std(magnitudes[j])))
		leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
		leg.get_frame().set_alpha(0.5)
		plt.subplot(212)
		plt.plot(times, phases[j], colors[j], markersize = markersize, label = '%s [avg = %0.5f, std = %0.5f]' % (channels[j], numpy.mean(phases[j]), numpy.std(phases[j])))
		leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
		leg.get_frame().set_alpha(0.5)
	plt.savefig("deltal_over_pcal_at_%0.1fHz_%d.png" % (frequencies[i], options.gps_start_time))

