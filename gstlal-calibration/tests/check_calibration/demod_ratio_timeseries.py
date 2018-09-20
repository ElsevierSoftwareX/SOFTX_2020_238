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
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--denominator-frame-cache", metavar = "name", type = str, help = "Frame cache file that contains denominator")
parser.add_option("--numerator-frame-cache", metavar = "name", type = str, help = "Frame cache file that contains numerator")
parser.add_option("--denominator-channel-name", metavar = "name", type = str, default = "CAL-PCALY_TX_PD_OUT_DQ", help = "Channel-name of denominator")
parser.add_option("--numerator-channel-name", metavar = "name", type = str, default = None, help = "Channel-name of numerator")
parser.add_option("--frequencies", metavar = "list", type = str, help = "List of frequencies at which to take ratios. Semicolons separate frequencies to be put on separate plots, and commas separate frequencies to be put on the same plot.")
parser.add_option("--filter-time", metavar = "seconds", type = int, default = 20, help = "Length in seconds of the low-pass filter used for demodulation")
parser.add_option("--average-time", metavar = "seconds", type = int, default = 1, help = "Length in seconds of the running average applied to the ratio")
parser.add_option("--magnitude-ranges", metavar = "list", type = str, help = "List of limits for magnitude plots. Semicolons separate ranges for different plots, and commas separate the min and max of a single plot.")
parser.add_option("--phase-ranges", metavar = "list", type = str, help = "List of limits for phase plots. Semicolons separate ranges for different plots, and commas separate the min and max of a single plot.")
parser.add_option("--plot-titles", metavar = "names", type = str, help = "Semicolon-separated list of titles for plots")

options, filenames = parser.parse_args()

ifo = options.ifo

# Set up channel list
channel_list = [(ifo, options.denominator_channel_name), (ifo, options.numerator_channel_name)]

# Convert the list of frequencies to a list of floats
freq_list = options.frequencies.split(';')
frequencies = []
for i in range(0, len(freq_list)):
	freq_list[i] = freq_list[i].split(',')
	for j in range(0, len(freq_list[i])):
		freq_list[i][j] = float(freq_list[i][j])
		frequencies.append(freq_list[i][j])

# demodulation and averaging parameters
filter_time = options.filter_time
average_time = options.average_time
rate_out = 1

#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


def demod_ratio(pipeline, name):

	# Get denominator data from the raw frames
	denominator_data = pipeparts.mklalcachesrc(pipeline, location = options.denominator_frame_cache, cache_dsc_regex = ifo)
	denominator_data = pipeparts.mkframecppchanneldemux(pipeline, denominator_data, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	denominator = calibration_parts.hook_up(pipeline, denominator_data, options.denominator_channel_name, ifo, 1.0)
	denominator = calibration_parts.caps_and_progress(pipeline, denominator, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", "denominator")
	denominator = pipeparts.mktee(pipeline, denominator)

	# Get numerator data from the raw frames
	if not options.denominator_frame_cache == options.numerator_frame_cache:
		numerator_data = pipeparts.mklalcachesrc(pipeline, location = options.numerator_frame_cache, cache_dsc_regex = ifo)
		numerator_data = pipeparts.mkframecppchanneldemux(pipeline, numerator_data, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
		numerator = calibration_parts.hook_up(pipeline, numerator_data, options.numerator_channel_name, ifo, 1.0)
	else:
		numerator = calibration_parts.hook_up(pipeline, denominator_data, options.numerator_channel_name, ifo, 1.0)
	numerator = calibration_parts.caps_and_progress(pipeline, numerator, "audio/x-raw,format=F64LE,channels=1,channel-mask=(bitmask)0x0", "numerator")
	numerator = pipeparts.mktee(pipeline, numerator)

	# Demodulate numerator and denominator at each frequency, take ratios, and write to file
	for i in range(0, len(frequencies)):
		# Demodulate
		demodulated_denominator = calibration_parts.demodulate(pipeline, denominator, frequencies[i], True, rate_out, filter_time, 0.5)
		demodulated_numerator = calibration_parts.demodulate(pipeline, numerator, frequencies[i], True, rate_out, filter_time, 0.5)
		demodulated_numerator = pipeparts.mktee(pipeline, demodulated_numerator)
		demodulated_denominator = pipeparts.mktee(pipeline, demodulated_denominator)
		# Take ratio
		ratio = calibration_parts.complex_division(pipeline, demodulated_numerator, demodulated_denominator)
		# Average
		ratio = pipeparts.mkgeneric(pipeline, ratio, "lal_smoothkappas", array_size = 1, avg_array_size = int(rate_out * average_time))
		# Find magnitude and phase
		ratio = pipeparts.mktee(pipeline, ratio)
		magnitude = pipeparts.mkgeneric(pipeline, ratio, "cabs")
		phase = pipeparts.mkgeneric(pipeline, ratio, "carg")
		phase = pipeparts.mkaudioamplify(pipeline, phase, 180.0 / numpy.pi)
		# Interleave
		magnitude_and_phase = calibration_parts.mkinterleave(pipeline, [magnitude, phase])
		# Write to file
		pipeparts.mknxydumpsink(pipeline, magnitude_and_phase, "%s_%s_over_%s_%0.1fHz.txt" % (ifo, options.numerator_channel_name, options.denominator_channel_name, frequencies[i]))

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
test_common.build_and_run(demod_ratio, "demod_ratio", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))

# Read data from files and plot it
colors = ['b.', 'y.', 'r.', 'c.', 'm.', 'g.'] # Hopefully the user will not want to plot more than six datasets on one plot.
for i in range(0, len(freq_list)):
	data = numpy.loadtxt("%s_%s_over_%s_%0.1fHz.txt" % (ifo, options.numerator_channel_name, options.denominator_channel_name, freq_list[i][0]))
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
	plt.plot(times, magnitudes[0], colors[0], markersize = markersize, label = '%0.1f Hz [avg = %0.5f, std = %0.5f]' % (freq_list[i][0], numpy.mean(magnitudes[0]), numpy.std(magnitudes[0])))
	plt.title(options.plot_titles.split(';')[i])
	plt.ylabel('Magnitude')
	magnitude_range = options.magnitude_ranges.split(';')[i]
	plt.ylim(float(magnitude_range.split(',')[0]), float(magnitude_range.split(',')[1]))
	plt.grid(True)
	leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.5)
	plt.subplot(212)
	plt.plot(times, phases[0], colors[0], markersize = markersize, label = '%0.1f Hz [avg = %0.5f, std = %0.5f]' % (freq_list[i][0], numpy.mean(phases[0]), numpy.std(phases[0])))
	plt.ylabel('Phase [deg]')
	plt.xlabel('Time in %s since %s UTC' % (t_unit, time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t_start + 315964782))))
	phase_range = options.phase_ranges.split(';')[i]
	plt.ylim(float(phase_range.split(',')[0]), float(phase_range.split(',')[1]))
	plt.grid(True)
	leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.5)
	for j in range(1, len(freq_list[i])):
		data = numpy.loadtxt("%s_%s_over_%s_%0.1fHz.txt" % (ifo, options.numerator_channel_name, options.denominator_channel_name, freq_list[i][j]))
		magnitudes.append([])
		phases.append([])
		for k in range(0, len(data)):
			magnitudes[j].append(data[k][1])
			phases[j].append(data[k][2])
		plt.subplot(211)
		plt.plot(times, magnitudes[j], colors[j], markersize = markersize, label = '%0.1f Hz [avg = %0.5f, std = %0.5f]' % (freq_list[i][j], numpy.mean(magnitudes[j]), numpy.std(magnitudes[j])))
		leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
		leg.get_frame().set_alpha(0.5)
		plt.subplot(212)
		plt.plot(times, phases[j], colors[j], markersize = markersize, label = '%0.1f Hz [avg = %0.5f, std = %0.5f]' % (freq_list[i][j], numpy.mean(phases[j]), numpy.std(phases[j])))
		leg = plt.legend(fancybox = True, markerscale = 4.0 / markersize, numpoints = 3)
		leg.get_frame().set_alpha(0.5)
	plt.savefig('%s_%d-%d.png' % (options.plot_titles.split(';')[i].replace(' ', '_'), int(t_start), int(dur)))

