#!/usr/bin/env python

import numpy
from math import pi
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import glob
from optparse import OptionParser, Option
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("--intime-file", metavar = "file", type = str, help = "File that contains data timestamps and real time of input data")
parser.add_option("--outtime-file", metavar = "file", type = str, help = "File that contains data timestamps and real time of output data")
parser.add_option("--plot-title", metavar = "name", type = str, help = "Title of the plot")
parser.add_option("--plot-filename-prefix", metavar = "file", type = str, help = "Start of the name of the file containing the plot. GPS start time, duration of plot, and .pdf are added")

options, filenames = parser.parse_args()

intimes = numpy.loadtxt(options.intime_file)
outtimes = numpy.loadtxt(options.outtime_file)

in_dt = intimes[1][0] - intimes[0][0]
out_dt = outtimes[1][0] - outtimes[0][0]

long_dt = max(in_dt, out_dt)
short_dt = min(in_dt, out_dt)
common_dt = long_dt
while common_dt % short_dt:
	common_dt = common_dt + long_dt

in_step = int(common_dt / in_dt)
out_step = int(common_dt / out_dt)

first_index_in = 0
first_index_out = 0
last_index_in = len(intimes) - 1
last_index_out = len(outtimes) - 1

while intimes[first_index_in][0] % common_dt:
	first_index_in = first_index_in + 1
while outtimes[first_index_out][0] % common_dt:
        first_index_out = first_index_out + 1
while intimes[last_index_in][0] % common_dt:
        last_index_in = last_index_in - 1
while outtimes[last_index_out][0] % common_dt:
        last_index_out = last_index_out - 1

while intimes[first_index_in][0] < outtimes[first_index_out][0]:
	first_index_in = first_index_in + in_step
while outtimes[first_index_out][0] < intimes[first_index_in][0]:
        first_index_out = first_index_out + out_step
while intimes[last_index_in][0] > outtimes[last_index_out][0]:
        last_index_in = last_index_in - in_step
while outtimes[last_index_out][0] > intimes[last_index_in][0]:
        last_index_out = last_index_out - out_step

t_start = intimes[first_index_in][0]
dur = intimes[last_index_in][0] - t_start
gps_time = []
latency = []
for i in range(0, 1 + (last_index_in - first_index_in) / in_step):
	gps_time.append(intimes[first_index_in + i * in_step][0] - t_start)
	latency.append(outtimes[first_index_out + i * out_step][1] - intimes[first_index_in + i * in_step][1])

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

plt.figure(figsize = (10, 5))
plt.plot(gps_time, latency, 'r.')
plt.title(options.plot_title)
plt.ylabel('Latency [s]')
plt.xlabel('Time in %s since %s UTC' % (t_unit, time.strftime("%b %d %Y %H:%M:%S", time.gmtime(t_start + 315964782))))
plt.ylim(0, 8)
plt.grid(True)
plt.savefig('%s_%d-%d.pdf' % (options.plot_filename_prefix, int(t_start), int(dur)))


