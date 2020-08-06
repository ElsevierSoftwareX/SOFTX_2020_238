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


import matplotlib; matplotlib.use('Agg')
import sys
import os
import numpy
import time
from math import pi
import resource
import datetime
import time
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 32
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['mathtext.default'] = 'regular'
import glob
import matplotlib.pyplot as plt
from ticks_and_grid import ticks_and_grid

from optparse import OptionParser, Option
import configparser

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
parser.add_option("--ifo", metavar = "name", help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--raw-frame-cache", metavar = "name", help = "Raw frame cache file")
parser.add_option("--gstlal-frame-cache-list", metavar = "list", help = "Comma-separated list of gstlal calibrated frame cache files to read")
parser.add_option("--config-file", metavar = "name", help = "Configurations file used to produce gstlal calibrated frames, needed to get pcal line frequencies and correction factors")
parser.add_option("--pcal-channel-name", metavar = "name", default = "CAL-PCALY_TX_PD_OUT_DQ", help = "Name of the pcal channel you wish to use")
parser.add_option("--gstlal-channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of gstlal calibrated channels to compare to pcal")
parser.add_option("--calcs-channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of gstlal calibrated channels in the raw frames to compare to pcal")
parser.add_option("--demodulation-time", metavar = "seconds", type = int, default = 128, help = "Time in seconds of low-pass filter used for demodulation. (Default = 128)")
parser.add_option("--pcal-line-names", metavar = "list", type = str, default = 'ka_pcal,kc_pcal,high_pcal', help = "Comma-separated list of pcal line names in filters file at which to plot ratios (default = 'ka_pcal,kc_pcal,high_pcal')")
parser.add_option("--magnitude-ranges", metavar = "list", type = str, default = "0.9,1.1;0.9,1.1;0.9,1.1", help = "Ranges for magnitude plots. Semicolons separate ranges for different plots, and commas separate min and max values.")
parser.add_option("--phase-ranges", metavar = "list", type = str, default = "-6.0,6.0;-6.0,6.0;-6.0,6.0", help = "Ranges for phase plots, in degrees. Semicolons separate ranges for different plots, and commas separate min and max values.")
parser.add_option("--labels", metavar = "list", type = str, help = "Comma-separated List of labels for each calibrated channel being tested. This is put in the plot legends and in the txt file names to distinguish them.")
parser.add_option("--file-name-suffix", metavar = "name", type = str, default = "", help = "Suffix for naming unique file.")
parser.add_option("--pcal-time-advance", metavar = "seconds", type = float, default = 0.0, help = "Time advance in seconds applied to the Pcal channel. Default = 0.0")
parser.add_option("--show-stats", action = "store_true", help = "If set, plots will have averages (mean) and standard deviations shown in plot legends.")

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

Config = configparser.ConfigParser()
Config.read(options.config_file)

InputConfigs = ConfigSectionMap("InputConfigurations")

#
# Load in the filters file that contains filter coefficients, etc.
#

# Search the directory tree for files with names matching the one we want.
filters_name = InputConfigs["filtersfilename"]
if filters_name.count('/') > 0:
	# Then the path to the filters file was given
	filters = numpy.load(filters_name)
else:
	# We need to search for the filters file
	filters_paths = []
	print("\nSearching for %s ..." % filters_name)
	# Check the user's home directory
	for dirpath, dirs, files in os.walk(os.environ['HOME']):
		if filters_name in files:
			# We prefer filters that came directly from a GDSFilters directory of the calibration SVN
			if dirpath.count("GDSFilters") > 0:
				filters_paths.insert(0, os.path.join(dirpath, filters_name))
			else:
				filters_paths.append(os.path.join(dirpath, filters_name))
	# Check if there is a checkout of the entire calibration SVN
	for dirpath, dirs, files in os.walk('/ligo/svncommon/CalSVN/aligocalibration/trunk/Runs/'):
		if filters_name in files:
			# We prefer filters that came directly from a GDSFilters directory of the calibration SVN
			if dirpath.count("GDSFilters") > 0:
				filters_paths.insert(0, os.path.join(dirpath, filters_name))
			else:
				filters_paths.append(os.path.join(dirpath, filters_name))
	if not len(filters_paths):
		raise ValueError("Cannot find filters file %s in home directory %s or in /ligo/svncommon/CalSVN/aligocalibration/trunk/Runs/*/GDSFilters", (filters_name, os.environ['HOME']))
	print("Loading calibration filters from %s\n" % filters_paths[0])
	filters = numpy.load(filters_paths[0])

ifo = options.ifo

# Set up gstlal frame cache list
gstlal_frame_cache_list = options.gstlal_frame_cache_list.split(',')

# Set up channel lists
channel_list = []
channel_list.append((ifo, options.pcal_channel_name))
if options.gstlal_channel_list is not None:
	gstlal_channels = options.gstlal_channel_list.split(',')
	for channel in gstlal_channels:
		channel_list.append((ifo, channel))
else:
	gstlal_channels = []
if options.calcs_channel_list is not None:
	calcs_channels = options.calcs_channel_list.split(',')
	for channel in calcs_channels:
		channel_list.append((ifo, channel))
else:
	calcs_channels = []

# Set up list of labels to be used in plot legends and filenames
labels = options.labels.split(',')

# Checks
if len(labels) != len(calcs_channels) + len(gstlal_channels):
	raise ValueError('Number of labels must equal number of channels (including calcs and gstlal channels) being measured. %d != %d' % (len(labels), len(calcs_channels) + len(gstlal_channels)))
if len(gstlal_frame_cache_list) != len(gstlal_channels):
	raise ValueError('Number of gstlal frame caches must equal number of gstlal channels. %d != %d' % (len(gstlal_frame_cache_list), len(gstlal_channels)))

# Read stuff we need from the filters file
frequencies = []
pcal_corrections = []
for name in options.pcal_line_names.split(','):
	frequencies.append(float(filters["%s_line_freq" % name]))
	pcal_corrections.append(float(filters["%s_corr_re" % name]))
	pcal_corrections.append(float(filters["%s_corr_im" % name]))
if(options.pcal_time_advance):
	for i in range(0, len(pcal_corrections) // 2):
		corr = pcal_corrections[2 * i] + 1j * pcal_corrections[2 * i + 1]
		corr *= numpy.exp(2.0 * numpy.pi * 1j * frequencies[i] * options.pcal_time_advance)
		pcal_corrections[2 * i] = numpy.real(corr)
		pcal_corrections[2 * i + 1] = numpy.imag(corr)

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
filter_time = options.demodulation_time
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
	raw_data = pipeparts.mkframecppchanneldemux(pipeline, raw_data, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))
	pcal = calibration_parts.hook_up(pipeline, raw_data, options.pcal_channel_name, ifo, 1.0)
	pcal = calibration_parts.caps_and_progress(pipeline, pcal, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", "pcal")
	pcal = pipeparts.mktee(pipeline, pcal)

	# Demodulate the pcal channel at the lines of interest
	for i in range(0, len(frequencies)):
		demodulated_pcal = calibration_parts.demodulate(pipeline, pcal, frequencies[i], True, rate_out, filter_time, 0.5, prefactor_real = pcal_corrections[2 * i], prefactor_imag = pcal_corrections[2 * i + 1])
		demodulated_pcal_list.append(pipeparts.mktee(pipeline, demodulated_pcal))

	# Check if we are taking pcal-to-darm ratios for CALCS data
	for channel, label in zip(calcs_channels, labels[0 : len(calcs_channels)]):
		calcs_deltal = calibration_parts.hook_up(pipeline, raw_data, channel, ifo, 1.0)
		calcs_deltal = calibration_parts.caps_and_progress(pipeline, calcs_deltal, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", label)
		calcs_deltal = pipeparts.mktee(pipeline, calcs_deltal)
		for i in range(0, len(frequencies)):
			# Demodulate DELTAL_EXTERNAL at each line
			demodulated_calcs_deltal = calibration_parts.demodulate(pipeline, calcs_deltal, frequencies[i], True, rate_out, filter_time, 0.5)
			# Take ratio \DeltaL(f) / pcal(f)
			deltaL_over_pcal = calibration_parts.complex_division(pipeline, demodulated_calcs_deltaL, demodulated_pcal_list[i])
			# Take a running average
			deltaL_over_pcal = pipeparts.mkgeneric(pipeline, deltaL_over_pcal, "lal_smoothkappas", array_size = 1, avg_array_size = int(rate_out * average_time))
			# Write to file
			pipeparts.mknxydumpsink(pipeline, deltaL_over_pcal, "%s_%s_over_%s_at_line%d.txt" % (ifo, label.replace(' ', '_'), options.pcal_channel_name, i + 1))

	# Check if we are taking pcal-to-darm ratios for gstlal calibrated data
	if options.gstlal_channel_list is not None:
		cache_num = 0
		for cache, channel, label in zip(gstlal_frame_cache_list, gstlal_channels, labels[len(calcs_channels) : len(channel_list)]):
			# Get gstlal channels from the gstlal frames
			hoft_data = pipeparts.mklalcachesrc(pipeline, location = cache, cache_dsc_regex = ifo)
			hoft_data = pipeparts.mkframecppchanneldemux(pipeline, hoft_data, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))
			hoft = calibration_parts.hook_up(pipeline, hoft_data, channel, ifo, 1.0, element_name_suffix = "_%d" % cache_num)
			cache_num = cache_num + 1
			hoft = calibration_parts.caps_and_progress(pipeline, hoft, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", label)
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
				pipeparts.mknxydumpsink(pipeline, magnitude_and_phase, "%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, label.replace(' ', '_'), options.pcal_channel_name, frequencies[i], options.gps_start_time))

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
colors = ['tomato', 'green', 'mediumblue', 'c', 'b', 'm'] # Hopefully the user will not want to plot more than six datasets on one plot.
channels = calcs_channels
channels.extend(gstlal_channels)
for i in range(0, len(frequencies)):
	data = numpy.loadtxt("%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, labels[0].replace(' ', '_'), options.pcal_channel_name, frequencies[i], options.gps_start_time))
	t_start = data[0][0]
	dur = data[len(data) - 1][0] - t_start
	dur_in_seconds = dur
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
	for k in range(0, int(len(data) / filter_time)):
		times.append((data[filter_time * k][0] - t_start) / sec_per_t_unit)
		magnitudes[0].append(data[filter_time * k][1])
		phases[0].append(data[filter_time * k][2])
	markersize = 150.0 * numpy.sqrt(float(filter_time / dur))
	markersize = min(markersize, 10.0)
	markersize = max(markersize, 0.2)
	# Make plots
	if i == 0:
		plt.figure(figsize = (25, 15))
	plt.subplot(2, len(frequencies), i + 1)
	if options.show_stats:
		plt.plot(times, magnitudes[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s} \ [\mu = %0.3f, \sigma = %0.3f]$' % (labels[0].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_'), numpy.mean(magnitudes[0]), numpy.std(magnitudes[0])))
	else:
		plt.plot(times, magnitudes[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s}$' % (labels[0].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_')))
	plt.title(r'${\rm %s} \ \widetilde{\Delta L}_{\rm free} / \tilde{x}_{\rm pc} \  {\rm at \  %0.1f \  Hz}$' % ( ifo, frequencies[i]), fontsize = 32)
	if i == 0:
		plt.ylabel(r'${\rm Magnitude}$')
	magnitude_range = options.magnitude_ranges.split(';')[i]
	ticks_and_grid(plt.gca(), ymin = float(magnitude_range.split(',')[0]), ymax = float(magnitude_range.split(',')[1]))
	leg = plt.legend(fancybox = True, markerscale = 8.0 / markersize, numpoints = 3, loc = 'upper right' if i < 2 else 'lower right')
	leg.get_frame().set_alpha(0.8)
	plt.subplot(2, len(frequencies), len(frequencies) + i + 1)
	if options.show_stats:
		plt.plot(times, phases[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s} \ [\mu = %0.2f^{\circ}, \sigma = %0.2f^{\circ}]$' % (labels[0].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_'), numpy.mean(phases[0]), numpy.std(phases[0])))
	else:
		plt.plot(times, phases[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s}$' % (labels[0].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_')))
	leg = plt.legend(fancybox = True, markerscale = 8.0 / markersize, numpoints = 3, loc = 'upper right' if i < 2 else 'lower right')
	leg.get_frame().set_alpha(0.8)
	if i == 0:
		plt.ylabel(r'${\rm Phase \  [deg]}$')
	if len(frequencies) < 3 or i == int((len(frequencies) - 0.1) / 2.0):
		plt.xlabel(r'${\rm Time \  in \  %s \  since \  %s \  UTC}$' % (t_unit, time.strftime("%b %d %Y %H:%M:%S".replace(':', '{:}').replace('-', '\mbox{-}').replace(' ', '\ '), time.gmtime(t_start + 315964782))))
	phase_range = options.phase_ranges.split(';')[i]
	ticks_and_grid(plt.gca(), ymin = float(phase_range.split(',')[0]), ymax = float(phase_range.split(',')[1]))
	for j in range(1, len(channels)):
		data = numpy.loadtxt("%s_%s_over_%s_at_%0.1fHz_%d.txt" % (ifo, labels[j].replace(' ', '_'), options.pcal_channel_name, frequencies[i], options.gps_start_time))
		magnitudes.append([])
		phases.append([])
		for k in range(0, int(len(data) / filter_time)):
			magnitudes[j].append(data[filter_time * k][1])
			phases[j].append(data[filter_time * k][2])
		plt.subplot(2, len(frequencies), i + 1)
		if options.show_stats:
			plt.plot(times, magnitudes[j], colors[j % 6], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s} \ [\mu = %0.3f, \sigma = %0.3f]$' % (labels[j].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_').replace(' ', '\ '), numpy.mean(magnitudes[j]), numpy.std(magnitudes[j])))
		else:
			plt.plot(times, magnitudes[j], colors[j % 6], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s}$' % (labels[j].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_').replace(' ', '\ ')))
		leg = plt.legend(fancybox = True, markerscale = 8.0 / markersize, numpoints = 3, loc = 'upper right' if i < 2 else 'lower right')
		leg.get_frame().set_alpha(0.8)
		plt.subplot(2, len(frequencies), len(frequencies) + i + 1)
		if options.show_stats:
			plt.plot(times, phases[j], colors[j % 6], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s} \ [\mu = %0.2f^{\circ}, \sigma = %0.2f^{\circ}]$' % (labels[j].replace(':', '{:}').replace('-' , '\mbox{-}').replace('_', '\_').replace(' ', '\ '), numpy.mean(phases[j]), numpy.std(phases[j])))
		else:
			plt.plot(times, phases[j], colors[j % 6], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s}$' % (labels[j].replace(':', '{:}').replace('-' , '\mbox{-}').replace('_', '\_').replace(' ', '\ ')))
		leg = plt.legend(fancybox = True, markerscale = 8.0 / markersize, numpoints = 3, loc = 'upper right' if i < 2 else 'lower right')
		leg.get_frame().set_alpha(0.8)
plt.savefig("%s_deltal_over_pcal%s_%d-%d.png" % (ifo, options.file_name_suffix, int(t_start), int(dur_in_seconds)))
plt.savefig("%s_deltal_over_pcal%s_%d-%d.pdf" % (ifo, options.file_name_suffix, int(t_start), int(dur_in_seconds)))

