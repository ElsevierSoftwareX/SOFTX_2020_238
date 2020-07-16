#!/usr/bin/env python3
# Copyright (C) 2019  Aaron Viets
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
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

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
parser.add_option("--frame-cache-list", metavar = "name", type = str, help = "Comma-separated list of frame cache files that contain data you want to plot")
parser.add_option("--channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of channel-names of which to compute RMS")
parser.add_option("--labels", metavar = "list", type = str, help = "Comma-separated list of labels for each channel. This is put in the plot legends and in the txt file names to distinguish them.")
parser.add_option("--sample-rates", metavar = "list", type = str, help = "Comma-separated list of sample rates for each channel.")
parser.add_option("--average-time", type = int, default = 30, help = "Time in seconds over which to take a running mean for the RMS")
parser.add_option("--fmin", type = float, default = None, help = "Minimum frequency for RMS")
parser.add_option("--fmax", type = float, default = None, help = "Maximum frequency for RMS")
parser.add_option("--plot-min", type = float, default = None, help = "Minimum for y-axis of plot")
parser.add_option("--plot-max", type = float, default = None, help = "Maximum for y-axis of plot")
parser.add_option("--filename-suffix", type = str, default = "", help = "Suffix for filename to make it unique.")

options, filenames = parser.parse_args()

# Get frame cache list, channel list, and labels
frame_cache_list = options.frame_cache_list.split(',')
channel_list = options.channel_list.split(',')
labels = options.labels.split(',')
rates = options.sample_rates.split(',')
for i in range(len(rates)):
	rates[i] = int(rates[i])
if len(frame_cache_list) != len(channel_list):
	raise ValueError('Number of frame cache files must equal number of channels: %d != %d' % (len(frame_cache_list), len(channel_list)))
if len(labels) != len(channel_list):
	raise ValueError('Number of labels must equal number of channels: %d != %d' % (len(labels), len(channel_list)))
if len(rates) != len(channel_list):
	raise ValueError('Number of sample rates must equal number of channels: %d != %d' % (len(rates), len(channel_list)))

# Read other options
ifo = options.ifo
temp = []
for i in range(0, len(channel_list)):
	temp.append((ifo, channel_list[i]))
channel_list = temp

gps_start_time = options.gps_start_time
gps_end_time = options.gps_end_time
data_duration = gps_end_time - gps_start_time
average_time = options.average_time
fmin = options.fmin
fmax = options.fmax
plot_min = options.plot_min
plot_mas = options.plot_max
filename_suffix = options.filename_suffix


# 
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


#
# This pipeline reads in specified channels, computes the RMS,
# and writes it to a txt file.
#

def compute_RMS_timeseries(pipeline, name):
	# Get the data from the frames
	for i in range(len(channel_list)):
		data = pipeparts.mklalcachesrc(pipeline, location = frame_cache_list[i], cache_dsc_regex = ifo)
		data = pipeparts.mkframecppchanneldemux(pipeline, data, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))
		data = calibration_parts.hook_up(pipeline, data, channel_list[i][1], ifo, 1.0)
		data = calibration_parts.caps_and_progress(pipeline, data, "audio/x-raw,format=F64LE", labels[i])
		RMS = calibration_parts.compute_rms(pipeline, data, rates[i], average_time, filter_length = 10.0, f_min = fmin, f_max = fmax, rate_out = 1)
		pipeparts.mknxydumpsink(pipeline, RMS, "%s_RMS_%s_%d-%d.txt" % (ifo, labels[i].replace(' ', '_'), gps_start_time, data_duration))

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
test_common.build_and_run(compute_RMS_timeseries, "compute_RMS_timeseries", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * gps_start_time), LIGOTimeGPS(0, 1000000000 * gps_end_time))))

colors = ['blue', 'limegreen', "royalblue", "deepskyblue", "red", "yellow", "purple", "pink"]

# Read data from txt file.
rms_data = []
tdata = []
for i in range(0, len(labels)):
	data = numpy.loadtxt('%s_RMS_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_'), gps_start_time, data_duration))
	if not any(tdata):
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
		for k in range(0, int(len(data) / average_time)):
			tdata.append((data[average_time * k][0] - t_start) / sec_per_t_unit)
		markersize = 150.0 * numpy.sqrt(float(average_time / dur))
		markersize = min(markersize, 10.0)
		markersize = max(markersize, 1.0)

	rms_data.append([])
	for k in range(0, int(len(data) / average_time)):
		rms_data[i].append(data[average_time * k][1])

for i in range(0, len(labels)):
	# Make plot
	if i == 0:
		plt.figure(figsize = (10, 6))
	plt.plot(tdata, rms_data[i], colors[i % 6], linestyle = 'None', marker = '.', markersize = markersize, label = labels[i].replace('_', '\_'))
	leg = plt.legend(fancybox = True, markerscale = 8.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.8)
	plt.gca().set_xscale('linear')
	plt.gca().set_yscale('log')
	if i == 0:
		if fmin is not None and fmax is not None:
			plt.title('%s Band-limited RMS [%d - %d Hz]' % (ifo, int(fmin), int(fmax)))
		elif fmin is not None:
			plt.title('%s RMS above %d Hz' % (ifo, int(fmin)))
		elif fmax is not None:
			plt.title('%s RMS below %d Hz' % (ifo, int(fmax)))
		else:
			plt.title('%s RMS' % ifo)
		plt.ylabel('RMS')
		plt.xlabel('Time [%s] from %s UTC' % (t_unit, time.strftime("%b %d %Y %H:%M:%S", time.gmtime(gps_start_time + 315964782))))
	if plot_min is not None:
		plt.ylim(plot_min, plot_max)
	plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')
plt.savefig('%s_RMS%s_%d-%d.png' % (ifo, filename_suffix, gps_start_time, data_duration))
plt.savefig('%s_RMS%s_%d-%d.pdf' % (ifo, filename_suffix, gps_start_time, data_duration))



