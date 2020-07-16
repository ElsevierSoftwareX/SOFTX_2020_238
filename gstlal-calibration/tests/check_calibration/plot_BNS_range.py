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
from math import pi
import resource
import datetime
import time

from gwpy.timeseries import TimeSeries
from gwpy.astro import inspiral_range

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

from optparse import OptionParser, Option


parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--frame-cache-list", metavar = "list", type = str, help = "Comma-separated list of frame cache files containing strain data")
parser.add_option("--channel-list", metavar = "list", type = str, help = "Comma-separated list of channel names for strain data")
parser.add_option("--integration-time", metavar = "seconds", type = int, default = 64, help = "Amount of data in seconds to use for each calculation of range")
parser.add_option("--stride", metavar = "seconds", type = int, default = 32, help = "Time-separation in seconds between consecutive values of range")
parser.add_option("--range-min", metavar = "Mpc", type = float, default = 0.0, help = "Minimum value for range on plot")
parser.add_option("--range-max", metavar = "Mpc", type = float, default = 160.0, help = "Maximum value for range on plot")
parser.add_option("--make-title", action = "store_true", help = "If set, a title will be added to the BNS range plot.")

options, filenames = parser.parse_args()

# parameters
gps_start_time = options.gps_start_time
gps_end_time = options.gps_end_time
ifo = options.ifo
integration_time = options.integration_time
stride = options.stride
range_min = options.range_min
range_max = options.range_max

num_points = int((gps_end_time - gps_start_time - integration_time + stride) / stride)

# Re-format list of frame caches and channels
frame_cache_list = options.frame_cache_list.split(',')
channel_list = options.channel_list.split(',')
if len(frame_cache_list) != len(channel_list):
	raise ValueError("--frame-cache-list and --channel-list must be the same length.")

# Make a time vector
times = []
for i in range(0, num_points):
	times.append(i * stride + 0.5 * integration_time)

# Decide what unit ot time to use in the plot
dur = times[-1]
if dur > 60 * 60 * 100:
	t_unit = 'days'
	sec_per_t_unit = 60.0 * 60.0 * 24.0
elif dur > 60 * 100:
	t_unit = 'hours'
	sec_per_t_unit = 60.0 * 60.0
elif dur > 100:
	t_unit = 'minutes'
	sec_per_t_unit = 60.0
else:
	t_unit = 'seconds'
	sec_per_t_unit = 1.0

for i in range(0, num_points):
	times[i] = times[i] / sec_per_t_unit

# Collect range data in arrays
ranges = []
medians = []
stds = []
for i in range(0, len(channel_list)):
	ranges.append([])
	for j in range(0, num_points):
		data = TimeSeries.read(frame_cache_list[i], "%s:%s" % (ifo, channel_list[i]), start = gps_start_time + j * stride, end = gps_start_time + j * stride + integration_time)
		PSD = data.psd(8, 4, method = 'lal_median')
		BNS_range = float(inspiral_range(PSD, fmin=10).value)
		ranges[i].append(BNS_range)
	medians.append(numpy.median(ranges[i]))
	stds.append(numpy.std(ranges[i]))
# Make plots
colors = ["blue", "green", "limegreen", "red", "yellow", "purple", "pink"] # Hopefully the user will not want to plot more than 7 datasets on one plot.
plt.figure(figsize = (12, 8))
for i in range(0, len(channel_list)):
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.plot(times, ranges[i], colors[i % 6], linewidth = 1.5, label = r'%s:%s [median = %0.1f Mpc, $\sigma$ = %0.1f Mpc]' % (ifo, channel_list[i].replace('_', '\_'), medians[i], stds[i]))
	if options.make_title:
		plt.title("%s binary neutron star inspiral range" % ifo)
	plt.ylabel('Angle-averaged range [Mpc]')
	plt.xlabel('Time [%s] from %s UTC' % (t_unit, time.strftime("%b %d %Y %H:%M:%S", time.gmtime(gps_start_time + 315964782))))
	plt.ylim(range_min, range_max)
	plt.grid(True)
	leg = plt.legend(fancybox = True)
	leg.get_frame().set_alpha(0.8)
plt.savefig('%s_BNS_range_%d-%d.png' % (ifo, int(gps_start_time), int(dur)))
plt.savefig('%s_BNS_range_%d-%d.pdf' % (ifo, int(gps_start_time), int(dur)))

