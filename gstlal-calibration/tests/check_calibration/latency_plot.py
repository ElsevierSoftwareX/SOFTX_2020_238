#!/usr/bin/env python3

import matplotlib; matplotlib.use('Agg')
import numpy
from math import pi
import datetime
import time
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import glob
from optparse import OptionParser, Option
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("--intime-file-list", metavar = "list", type = str, help = "Comma-separated list of files that contain data timestamps and real times of input data")
parser.add_option("--outtime-file-list", metavar = "file", type = str, help = "Comma-separated list of files that contain data timestamps and real times of output data")
parser.add_option("--labels", metavar = "list", type = str, default = "", help = "Comma-separated list of plot legends for the data sets (default is no legend)")
parser.add_option("--plot-title", metavar = "name", type = str, default = "", help = "Title of the plot (default is no title)")
parser.add_option("--plot-filename-prefix", metavar = "file", type = str, default = "", help = "Start of the name of the file containing the plot. GPS start time, duration of plot, and .pdf are added")

options, filenames = parser.parse_args()

labels = options.labels.split(',')

# Get the list of files with the timestamp data
intime_file_list = options.intime_file_list.split(',')
outtime_file_list = options.outtime_file_list.split(',')
if len(intime_file_list) != len(outtime_file_list):
	raise ValueError("intime-file-list and outtime-file-list must be the same length")

# Organize the times into lists
intimes = []
outtimes = []
for i in range(0, len(intime_file_list)):
	intimes.append(numpy.loadtxt(intime_file_list[i]))
	outtimes.append(numpy.loadtxt(outtime_file_list[i]))

# Find the least common multiple sample period.  This assumes that all input
# sample periods are the same and all output sample periods are the same.
in_dt = intimes[0][1][0] - intimes[0][0][0]
out_dt = outtimes[0][1][0] - outtimes[0][0][0]
long_dt = max(in_dt, out_dt)
short_dt = min(in_dt, out_dt)
common_dt = long_dt
while common_dt % short_dt:
	common_dt = common_dt + long_dt

# Number of samples of input and output corresponding to least common multiple sample period
in_step = round(common_dt / in_dt)
out_step = round(common_dt / out_dt)

# Sometimes the incoming data has dropouts... we need to deal with this :(
for i in range(0, len(intimes)):
	j = 1
	while j < len(intimes[i]):
		if round((intimes[i][j][0] - intimes[i][j - 1][0]) / in_dt) > 1:
			intimes[i] = numpy.insert(intimes[i], j, [intimes[i][j - 1][0] + in_dt, intimes[i][j - 1][1] + in_dt], axis = 0)
		j = j + 1

# Find a start time that is not before any of the data sets start and an end time that is not after any of the data sets end
t_start = intimes[0][0][0]
t_end = intimes[0][-1][0]
for i in range(0, len(intimes)):
	t_start = t_start if t_start > intimes[i][0][0] else intimes[i][0][0]
	t_start = t_start if t_start > outtimes[i][0][0] else outtimes[i][0][0]
	t_end = t_end if t_end < intimes[i][-1][0] else intimes[i][-1][0]
	t_end = t_end if t_end < outtimes[i][-1][0] else outtimes[i][-1][0]

# Make sure the start and end times are multiples of the chosen sample period
if t_start % common_dt:
	t_start = t_start + common_dt - t_start % common_dt
if t_end % common_dt:
	t_end = t_end - t_end % common_dt

# Make a time vector
dur = t_end - t_start
gps_time = numpy.arange(0, dur + common_dt / 2, common_dt)

# Decide what unit of time to use
t_unit = 'seconds'
if gps_time[len(gps_time) - 1] > 100:
	for i in range(0, len(gps_time)):
		gps_time[i] = gps_time[i] / 60.0
	t_unit = 'minutes'
	if gps_time[len(gps_time) - 1] > 100:
		for i in range(0, len(gps_time)):
			gps_time[i] = gps_time[i] / 60.0
		t_unit = 'hours'
		if gps_time[len(gps_time) - 1] > 100:
			for i in range(0, len(gps_time)):
				gps_time[i] = gps_time[i] / 24.0
			t_unit = 'days'

# Collect latency data in a list
latency = []
for i in range(0, len(intimes)):
	latency.append([])
	intimes_start_index = round((t_start - intimes[i][0][0]) / in_dt)
	outtimes_start_index = round((t_start - outtimes[i][0][0]) / out_dt)
	for j in range(0, len(gps_time)):
		latency[i].append(outtimes[i][outtimes_start_index + j * out_step][1] - intimes[i][intimes_start_index + j * in_step][1])

# Make the plot
colors = ['blue', 'limegreen', 'orchid', 'cyan', 'g', 'm', 'y', 'r']
markersize = 150.0 / numpy.sqrt(len(intimes[0]))
markersize = min(markersize, 8.0)
markersize = max(markersize, 1.0)
plt.figure(figsize = (10, 6))
if len(labels[0]):
	plt.plot(gps_time, latency[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s \ }[\mu_{1/2} = %0.2f \, {\rm s}, \sigma = %0.2f \, {\rm s}]$' % (labels[0].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_').replace(' ', '\ '), numpy.median(latency[0]), numpy.std(latency[0])))
	leg = plt.legend(fancybox = True, loc = 'upper right', markerscale = 8.0 / markersize, numpoints = 3)
	leg.get_frame().set_alpha(0.8)
else:
	plt.plot(gps_time, latency[0], colors[0], linestyle = 'None', marker = '.', markersize = markersize)
if len(options.plot_title):
	plt.title(options.plot_title)
plt.ylabel(r'${\rm Latency \ [s]}$')
plt.xlabel(r'${\rm Time \ in \ %s \ since \ %s \ UTC}$' % (t_unit, time.strftime("%b %d %Y %H:%M:%S".replace(':', '{:}').replace('-', '\mbox{-}').replace(' ', '\ '), time.gmtime(t_start + 315964782))))
plt.ylim(0, 5)
plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')

for i in range(1, len(intimes)):
	if len(labels) > 1:
		plt.plot(gps_time, latency[i], colors[i % len(colors)], linestyle = 'None', marker = '.', markersize = markersize, label = r'${\rm %s \ }[\mu_{1/2} = %0.2f \, {\rm s}, \sigma = %0.2f \, {\rm s}]$' % (labels[i].replace(':', '{:}').replace('-', '\mbox{-}').replace('_', '\_').replace(' ', '\ '), numpy.median(latency[i]), numpy.std(latency[i])))
		leg = plt.legend(fancybox = True, loc = 'upper right', markerscale = 8.0 / markersize, numpoints = 3)
		leg.get_frame().set_alpha(0.8)
	else:
		plt.plot(gps_time, latency[i], colors[i % len(colors)], linestyle = 'None', marker = '.', markersize = markersize)

# Save the plot to a file
plt.savefig('%s_%d-%d.png' % (options.plot_filename_prefix, int(t_start), int(dur)))
plt.savefig('%s_%d-%d.pdf' % (options.plot_filename_prefix, int(t_start), int(dur)))


