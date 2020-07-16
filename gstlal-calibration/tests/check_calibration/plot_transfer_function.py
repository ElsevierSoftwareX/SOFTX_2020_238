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
from math import pi
import resource
import datetime
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['mathtext.default'] = 'regular'
import glob
import matplotlib.pyplot as plt

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

from ticks_and_grid import ticks_and_grid

parser = OptionParser()
parser.add_option("--gps-start-time", metavar = "seconds", type = int, help = "GPS time at which to start processing data")
parser.add_option("--gps-end-time", metavar = "seconds", type = int, help = "GPS time at which to stop processing data")
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--denominator-frame-cache", metavar = "name", type = str, help = "Name of frame cache file that contains denominator of transfer functions")
parser.add_option("--denominator-channel-name", metavar = "name", type = str, default = None, help = "Channel-name of denominator")
parser.add_option("--denominator-name", metavar = "name", type = str, default = '{\\rm Pcal}(f)', help = "Name of denominator in plot title, in latex math mode")
parser.add_option("--denominator-correction", metavar = "name", type = str, default = None, help = "Name of filters-file parameter needed to apply a correction to the denominator")
parser.add_option("--numerator-frame-cache-list", metavar = "name", type = str, help = "Comma-separated list of frame cache files that contain numerators of transfer functions")
parser.add_option("--numerator-channel-list", metavar = "list", type = str, default = None, help = "Comma-separated list of channel-names of numerators")
parser.add_option("--numerator-name", metavar = "name", type = str, default = '\\Delta L_{\\rm free}(f)', help = "Name of numerator in plot title, in latex math mode")
parser.add_option("--numerator-correction", metavar = "name", type = str, default = None, help = "Name of filters-file parameter needed to apply a correction to the numerators.  If there are multiple different corrections corresponding to different numerator channels, pass as a comma-separated list (must be equal in length as the numerator-frame-cache-list and the numerator-channel-list)")
parser.add_option("--zeros", metavar = "list", type = str, default = None, help = "Comma-separated list of real and imaginary parts of zeros to filter the transfer function with.  Note that if you want to apply zeros to the denominator, you must apply poles to the transfer function.")
parser.add_option("--poles", metavar = "list", type = str, default = None, help = "Comma-separated list of real and imaginary parts of poles to filter the transfer function with.  Note that if you want to apply poles to the denominator, you must apply zeros to the transfer function.")
parser.add_option("--gain", type = float, default = 1.0, help = "Gain factor to apply to the transfer function")
parser.add_option("--filters-file", metavar = "name", default = None, help = "Filters file used to produce GDS/DCS calibrated frames.  Either this or config-file is needed if applying any correction to numerator or denominator")
parser.add_option("--config-file", metavar = "name", default = None, help = "Configurations file used to produce GDS/DCS calibrated frames.  Either this or filters-file is needed if applying any correction to numerator or denominator")
parser.add_option("--sample-rate", metavar = "Hz", type = int, default = 16384, help = "Sample rate at which transfer function is computed")
parser.add_option("--fft-time", metavar = "seconds", type = float, default = 16, help = "Length of FFTs used to compute transfer function")
parser.add_option("--use-median", action = "store_true", help = "Use a median instead of an average to compute transfer function")
parser.add_option("--df", metavar = "Hz", type = float, default = 0.25, help = "Frequency spacing of transfer function")
parser.add_option("--frequency-min", type = float, default = 10, help = "Minimum frequency for plot")
parser.add_option("--frequency-max", type = float, default = 5000, help = "Maximum frequency for plot")
parser.add_option("--magnitude-min", type = float, default = 0.9, help = "Minimum for magnitude plot")
parser.add_option("--magnitude-max", type = float, default = 1.1, help = "Maximum for magnitude plot")
parser.add_option("--phase-min", metavar = "degrees", type = float, default = -6, help = "Minimum for phase plot, in degrees")
parser.add_option("--phase-max", metavar = "degrees", type = float, default = 6, help = "Maximum for phase plot, in degrees")
parser.add_option("--labels", metavar = "list", type = str, default = None, help = "Comma-separated list of labels corresponding to each transfer function, to be added to plot legend and txt file names.")
parser.add_option("--filename-suffix", type = str, default = "", help = "Suffix for filename to make it unique.")
parser.add_option("--filename", type = str, default = None, help = "Name of plot filef, not including file extension.  Not required.")


options, filenames = parser.parse_args()

# Get numerator frame cache list and channel list
numerator_frame_cache_list = options.numerator_frame_cache_list.split(',')
numerator_channel_list = options.numerator_channel_list.split(',')
labels = options.labels.split(',')
if len(numerator_frame_cache_list) != len(numerator_channel_list):
	raise ValueError('Number of numerator frame cache files must equal number of numerator channels: %d != %d' % (len(numerator_frame_cache_list), len(numerator_channel_list)))
if len(labels) != len(numerator_channel_list):
	raise ValueError('Number of labels must equal number of numerator channels: %d != %d' % (len(labels), len(numerator_channel_list)))

# Read other options
ifo = options.ifo
channel_list = [(ifo, options.denominator_channel_name)]
for i in range(0, len(numerator_channel_list)):
	channel_list.append((ifo, numerator_channel_list[i]))
sample_rate = options.sample_rate
data_duration = options.gps_end_time - options.gps_start_time
tf_samples = data_duration * sample_rate
fft_length = int(options.fft_time * sample_rate)
tf_length = int(1 + sample_rate / (2 * options.df))
if not tf_length % 2:
	tf_length = tf_length + 1
td_tf_length = 2 * (tf_length - 1)
fft_overlap = fft_length // 2
num_ffts = int((tf_samples - fft_overlap) / (fft_length - fft_overlap))

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

filters_name = None
if options.filters_file is not None:
	filters_name = options.filters_file

elif options.config_file is not None:
	# Read the config file
	Config = configparser.ConfigParser()
	Config.read(options.config_file)

	InputConfigs = ConfigSectionMap("InputConfigurations")

	filters_name = InputConfigs["filtersfilename"]

if filters_name is not None:
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

# Get the corrections for numerators and denominator, and resample if necessary
num_corr = []
denom_corr = []
numerator_correction = options.numerator_correction
if numerator_correction == 'None':
	numerator_correction = None
denominator_correction = options.denominator_correction
if denominator_correction == 'None':
	denominator_correction = None

if numerator_correction is not None:
	numerator_correction = numerator_correction.split(',')
	for i in range(len(numerator_correction)):
		if numerator_correction[i] == 'None':
			numerator_correction[i] = None
	# Turn it into a list of identical strings if there is only one correction given
	if len(numerator_correction) < len(numerator_frame_cache_list):
		for i in range(len(numerator_frame_cache_list) - len(numerator_correction)):
			numerator_correction.append(numerator_correction[-1])
	for i in range(len(numerator_correction)):
		num_corr.append([])
		if numerator_correction[i] is not None and numpy.size(filters[numerator_correction[i]]) > 1:
			corr = filters[numerator_correction[i]]
			# Check the frequency spacing of the correction
			corr_df = corr[0][1] - corr[0][0]
			cadence = options.df / corr_df
			index = 0
			# This is a linear resampler (it just connects the dots with straight lines)
			while index < tf_length - 1:
				before_idx = int(numpy.floor(cadence * index))
				after_idx = int(numpy.ceil(cadence * index + 1e-10))
				before = corr[1][before_idx] + 1j * corr[2][before_idx]
				after = corr[1][after_idx] + 1j * corr[2][after_idx]
				before_weight = after_idx - cadence * index
				after_weight = cadence * index - before_idx
				num_corr[i].append(before_weight * before + after_weight * after)
				index += 1
			# Check if we can add the last value
			before_idx = int(numpy.floor(cadence * index))
			if numpy.floor(cadence * index) == cadence * index:
				num_corr[i].append(corr[1][before_idx] + 1j * corr[2][before_idx])

if denominator_correction is not None:
	if numpy.size(filters[denominator_correction]) > 1:
		corr = filters[denominator_correction]
		# Check the frequency spacing of the correction
		corr_df = corr[0][1] - corr[0][0]
		cadence = options.df / corr_df
		index = 0
		# This is a linear resampler (it just connects the dots with straight lines)
		while index < tf_length - 1:
			before_idx = int(numpy.floor(cadence * index))
			after_idx = int(numpy.ceil(cadence * index + 1e-10))
			before = corr[1][before_idx] + 1j * corr[2][before_idx]
			after = corr[1][after_idx] + 1j * corr[2][after_idx]
			before_weight = after_idx - cadence * index
			after_weight = cadence * index - before_idx
			if("PCALX" in options.denominator_channel_name):
				denom_corr.append(-1.0 * (before_weight * before + after_weight * after))
			else:
				denom_corr.append(before_weight * before + after_weight * after)
			index += 1
		# Check if we can add the last value
		before_idx = int(numpy.floor(cadence * index))
		if numpy.floor(cadence * index) == cadence * index:
			if("PCALX" in options.denominator_channel_name):
				denom_corr.append(-1.0 * (corr[1][before_idx] + 1j * corr[2][before_idx]))
			else:
				denom_corr.append(corr[1][before_idx] + 1j * corr[2][before_idx])

# 
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


#
# This pipeline reads in two specified channels from two (generally different)
# frame caches and computes and plots a transfer function.
#


def plot_transfer_function(pipeline, name):
	# Get the data from the denominator frames
	denominator = pipeparts.mklalcachesrc(pipeline, location = options.denominator_frame_cache, cache_dsc_regex = ifo)
	denominator = pipeparts.mkframecppchanneldemux(pipeline, denominator, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))
	denominator = calibration_parts.hook_up(pipeline, denominator, options.denominator_channel_name, ifo, 1.0)
	denominator = calibration_parts.caps_and_progress(pipeline, denominator, "audio/x-raw,format=F64LE", "denominator")
	denominator = calibration_parts.mkresample(pipeline, denominator, 5, False, int(sample_rate))
	if denominator_correction is not None:
		if numpy.size(filters[denominator_correction]) == 1:
			denominator = pipeparts.mkaudioamplify(pipeline, denominator, float(filters[denominator_correction]))
	denominator = pipeparts.mktee(pipeline, denominator)

	# Get the data from the numerator frames
	for i in range(0, len(labels)):
		numerator = pipeparts.mklalcachesrc(pipeline, location = numerator_frame_cache_list[i], cache_dsc_regex = ifo)
		numerator = pipeparts.mkframecppchanneldemux(pipeline, numerator, do_file_checksum = False, skip_bad_files = True, channel_list = list(map("%s:%s".__mod__, channel_list)))
		numerator = calibration_parts.hook_up(pipeline, numerator, numerator_channel_list[i], ifo, 1.0, element_name_suffix = "%d" % i)
		numerator = calibration_parts.caps_and_progress(pipeline, numerator, "audio/x-raw,format=F64LE", labels[i])
		numerator = calibration_parts.mkresample(pipeline, numerator, 5, False, int(sample_rate))
		if numerator_correction is not None:
			if numerator_correction[i] is not None:
				if numpy.size(filters[numerator_correction[i]] == 1):
					numerator = pipeparts.mkaudioamplify(pipeline, numerator, float(filters[numerator_correction[i]]))
		# Interleave the channels to make one stream
		channels = calibration_parts.mkinterleave(pipeline, [numerator, denominator])
		# Send the data to lal_transferfunction to compute and write transfer functions
		pipeparts.mkgeneric(pipeline, channels, "lal_transferfunction", fft_length = fft_length, fft_overlap = fft_overlap, num_ffts = num_ffts, fir_length = td_tf_length, use_median = True if options.use_median else False, update_samples = 1e15, filename = '%s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration))

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
test_common.build_and_run(plot_transfer_function, "plot_transfer_function", segment = segments.segment((LIGOTimeGPS(0, 1000000000 * options.gps_start_time), LIGOTimeGPS(0, 1000000000 * options.gps_end_time))))

# Get any zeros and poles that we want to use to filter the transfer functions
zeros = []
if options.zeros is not None:
	real_zeros = options.zeros.split(';')
	if len(real_zeros) < len(labels):
		# Then copy the last element until they are the same length
		for i in range(len(labels) - len(real_zeros)):
			real_zeros.append(real_zeros[-1])
	for i in range(len(real_zeros)):
		real_zeros[i] = real_zeros[i].split(',')
		zeros.append([])
		for j in range(0, len(real_zeros[i]) // 2):
			zeros[i].append(float(real_zeros[i][2 * j]) + 1j * float(real_zeros[i][2 * j + 1]))

poles = []
if options.poles is not None:
	real_poles = options.poles.split(';')
	if len(real_poles) < len(labels):
		# Then copy the last element until they are the same length
		for i in range(len(labels) - len(real_poles)):
			real_poles.append(real_poles[-1])
	for i in range(len(real_poles)):
		real_poles[i] = real_poles[i].split(',')
		poles.append([])
		for j in range(0, len(real_poles[i]) // 2):
			poles[i].append(float(real_poles[i][2 * j]) + 1j * float(real_poles[i][2 * j + 1]))

# Decide a color scheme.
colors = []

# Start with defaults to use if we don't recognize what we are plotting.
default_colors = ['red', 'gold', 'magenta', 'orange', 'aqua', 'darkgreen', 'blue']

# Use specific colors for known version of calibration
C02_labels = ['C02', 'c02']
C01_labels = ['C01', 'c01', 'DCS', 'dcs', 'high-latency', 'High-Latency', 'High-latency', 'HIGH-LATENCY', 'high_latency', 'High_Latency', 'High_latency', 'HIGH_LATENCY', 'high latency', 'High Latency', 'High latency', 'HIGH LATENCY', 'offline', 'Offline', 'OFFLINE']
C00_labels = ['C00', 'c00', 'GDS', 'gds', 'low-latency', 'Low-Latency', 'Low-latency', 'LOW-LATENCY', 'low_latency', 'Low_Latency', 'Low_latency', 'LOW_LATENCY', 'low latency', 'Low Latency', 'Low latency', 'LOW LATENCY', 'online', 'Online', 'ONLINE']
CALCS_labels = ['CALCS', 'calcs', 'Calcs', 'CalCS', 'CAL-CS', 'Cal-CS', 'cal-cs', 'CAL_CS', 'Cal_CS', 'cal_cs', 'front', 'FRONT', 'Front']
for i in range(len(labels)):
	if any ([x for x in C02_labels if x in labels[i]]):
		colors.append('springgreen')
	elif any ([x for x in C01_labels if x in labels[i]]):
		colors.append('maroon')
	elif any ([x for x in C00_labels if x in labels[i]]):
		colors.append('royalblue')
	elif any([x for x in CALCS_labels if x in labels[i]]):
		colors.append('silver')
	else:
		colors.append(default_colors[i % 7])

for i in range(0, len(labels)):
	# Remove unwanted lines from file, and re-format wanted lines
	f = open('%s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration),"r")
	lines = f.readlines()
	f.close()
	f = open('%s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration),"w")
	for j in range(3, 3 + tf_length):
		f.write(lines[j].replace(' + ', '\t').replace(' - ', '\t-').replace('i', ''))
	f.close()

	# Read data from re-formatted file and find frequency vector, magnitude, and phase
	data = numpy.loadtxt('%s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration))
	frequency = []
	magnitude = []
	phase = []
	for j in range(0, len(data)):
		frequency.append(data[j][0])
		tf_at_f = (data[j][1] + 1j * data[j][2]) * options.gain
		if len(num_corr[i]) > j:
			tf_at_f *= num_corr[i][j]
		if len(denom_corr) > j:
			tf_at_f /= denom_corr[j]
		for z in zeros[i]:
			tf_at_f = tf_at_f * (1.0 + 1j * frequency[j] / z)
		for p in poles[i]:
			tf_at_f = tf_at_f / (1.0 + 1j * frequency[j] / p)
		magnitude.append(abs(tf_at_f))
		phase.append(numpy.angle(tf_at_f) * 180.0 / numpy.pi)

	# Save txt file with the "corrected" data
	os.system('rm %s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration))
	if options.filename is not None:
		numpy.savetxt('%s.txt' % options.filename, numpy.transpose([frequency, magnitude, phase]), fmt = '%8e', delimiter = "   ", header = "Freq. (Hz)	Magnitude	Phase (deg)")
	else:
		numpy.savetxt('%s_%s_over_%s_%d-%d.txt' % (ifo, labels[i].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration), numpy.transpose([frequency, magnitude, phase]), fmt = '%8e', delimiter = "   ", header = "Freq. (Hz)	Magnitude	Phase (deg)")

	# Make plots
	freq_scale = 'log' if options.frequency_min > 0.0 and options.frequency_max / options.frequency_min > 10 else 'linear'
	mag_scale = 'log' if options.magnitude_min > 0.0 and options.magnitude_max / options.magnitude_min > 10 else 'linear'
	if i == 0:
		plt.figure(figsize = (10, 10))
	plt.subplot(211)
	plt.plot(frequency, magnitude, colors[i % 6], linewidth = 0.75, label = r'${\rm %s}$' % labels[i].replace('_', '\_').replace(' ', '\ '))
	leg = plt.legend(fancybox = True)
	leg.get_frame().set_alpha(0.8)
	if i == 0:
		plt.title(r'${\rm %s} \ %s \ / \ %s$' % (ifo, options.numerator_name, options.denominator_name))
		plt.ylabel(r'${\rm Magnitude}$')
	ticks_and_grid(plt.gca(), xmin = options.frequency_min, xmax = options.frequency_max, ymin = options.magnitude_min, ymax = options.magnitude_max, xscale = freq_scale, yscale = mag_scale)
	ax = plt.subplot(212)
	plt.plot(frequency, phase, colors[i % 6], linewidth = 0.75)
	if i == 0:
		plt.ylabel(r'${\rm Phase \ [deg]}$')
		plt.xlabel(r'${\rm Frequency \ [Hz]}$')
	ticks_and_grid(plt.gca(), xmin = options.frequency_min, xmax = options.frequency_max, ymin = options.phase_min, ymax = options.phase_max, xscale = freq_scale, yscale = 'linear')
if options.filename is not None:
	plt.savefig('%s.pdf' % options.filename)
	plt.savefig('%s.png' % options.filename)
else:
	plt.savefig('%s_%s_%s_over_%s_%d-%d.pdf' % (ifo, numerator_channel_list[-1], labels[-1].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration))
	plt.savefig('%s_%s_%s_over_%s_%d-%d.png' % (ifo, numerator_channel_list[-1], labels[-1].replace(' ', '_').replace('/', 'over'), options.denominator_channel_name, options.gps_start_time, data_duration))

