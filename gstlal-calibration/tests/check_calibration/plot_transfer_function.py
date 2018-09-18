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
parser.add_option("--ifo", metavar = "name", type = str, help = "Name of the interferometer (IFO), e.g., H1, L1")
parser.add_option("--denominator-frame-cache", metavar = "name", type = str, help = "Frame cache file that contains denominator")
parser.add_option("--numerator-frame-cache", metavar = "name", type = str, help = "Frame cache file that contains numerator")
parser.add_option("--denominator-channel-name", metavar = "name", type = str, default = None, help = "Channel-name of denominator")
parser.add_option("--numerator-channel-name", metavar = "name", type = str, default = None, help = "Channel-name of numerator")
parser.add_option("--denominator-correction", metavar = "name", type = str, default = None, help = "Name of filters-file parameter needed to apply a correction to the denominator")
parser.add_option("--numerator-correction", metavar = "name", type = str, default = None, help = "Name of filters-file parameter needed to apply a correction to the numerator")
parser.add_option("--denominator-correction-delay", metavar = "name", type = str, default = None, help = "Name of filters-file delay parameter needed to apply a correction to the denominator")
parser.add_option("--numerator-correction-delay", metavar = "name", type = str, default = None, help = "Name of filters-file delay parameter needed to apply a correction to the numerator")
parser.add_option("--config-file", metavar = "name", default = None, help = "Configurations file used to produce GDS/DCS calibrated frames, needed if applying any correction to numerator or denominator")
parser.add_option("--sample-rate", metavar = "Hz", type = int, default = 16384, help = "Sample rate at which transfer function is computed")
parser.add_option("--fft-time", metavar = "seconds", type = float, default = 16, help = "Length of FFTs used to compute transfer function")
parser.add_option("--use-median", action = "store_true", help = "Use a median instead of an average to compute transfer function")
parser.add_option("--df", metavar = "Hz", type = float, default = 0.25, help = "Frequency spacing of transfer function")
parser.add_option("--frequency-min", type = float, default = 10, help = "Minimum frequency for plot")
parser.add_option("--frequency-max", type = float, default = 5000, help = "Maximum frequency for plot")
parser.add_option("--magnitude-min", type = float, default = 0.01, help = "Minimum for magnitude plot")
parser.add_option("--magnitude-max", type = float, default = 100, help = "Maximum for magnitude plot")
parser.add_option("--phase-min", metavar = "degrees", type = float, default = -90, help = "Minimum for phase plot, in degrees")
parser.add_option("--phase-max", metavar = "degrees", type = float, default = 90, help = "Maximum for phase plot, in degrees")
parser.add_option("--plot-title", metavar = "name", type = str, help = "Title for plot")

options, filenames = parser.parse_args()

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

if options.config_file is not None:
	# Read the config file
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
channel_list = [(ifo, options.denominator_channel_name), (ifo, options.numerator_channel_name)]
sample_rate = options.sample_rate
data_duration = options.gps_end_time - options.gps_start_time
if options.numerator_correction_delay is not None and options.denominator_correction_delay is not None:
	chop_samples = max(len(filters[options.numerator_correction]), len(filters[options.denominator_correction]))
elif options.numerator_correction_delay is not None:
	chop_samples = len(filters[options.numerator_correction])
elif options.denominator_correction_delay is not None:
	chop_samples = len(filters[options.denominator_correction])
else:
	chop_samples = 0
tf_samples = data_duration * sample_rate - 2 * chop_samples
fft_length = int(options.fft_time * sample_rate)
tf_length = int(1 + sample_rate / options.df)
if not tf_length % 2:
	tf_length = tf_length + 1
td_tf_length = 2 * (tf_length - 1)
fft_overlap = fft_length / 2
num_ffts = int((tf_samples - fft_overlap) / (fft_length - fft_overlap))

channel_list = [(ifo, options.denominator_channel_name), (ifo, options.numerator_channel_name)]

#
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

	# Get the data from the frames
	numerator = pipeparts.mklalcachesrc(pipeline, location = options.numerator_frame_cache, cache_dsc_regex = ifo)
	numerator = pipeparts.mkframecppchanneldemux(pipeline, numerator, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	numerator = calibration_parts.hook_up(pipeline, numerator, options.numerator_channel_name, ifo, 1.0)
	numerator = calibration_parts.caps_and_progress(pipeline, numerator, "audio/x-raw,format=F64LE", "numerator")
	numerator = calibration_parts.mkresample(pipeline, numerator, 5, False, int(sample_rate))
	if options.numerator_correction is not None:
		numerator_correction = filters[options.numerator_correction]
		if options.numerator_correction_delay is None:
			numerator = pipeparts.mkaudioamplify(pipeline, numerator, float(numerator_correction))
		else:
			numerator = pipeparts.mkfirbank(pipeline, numerator, latency = int(filters[options.numerator_correction_delay]), fir_matrix = [numerator_correction[::-1]], time_domain = True)

	denominator = pipeparts.mklalcachesrc(pipeline, location = options.denominator_frame_cache, cache_dsc_regex = ifo)
	denominator = pipeparts.mkframecppchanneldemux(pipeline, denominator, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	denominator = calibration_parts.hook_up(pipeline, denominator, options.denominator_channel_name, ifo, 1.0)
	denominator = calibration_parts.caps_and_progress(pipeline, denominator, "audio/x-raw,format=F64LE", "denominator")
	denominator = calibration_parts.mkresample(pipeline, denominator, 5, False, int(sample_rate))
	if options.denominator_correction is not None:
		denominator_correction = filters[options.denominator_correction]
		if options.denominator_correction_delay is None:
			denominator = pipeparts.mkaudioamplify(pipeline, denominator, float(denominator_correction))
		else:
			denominator = pipeparts.mkfirbank(pipeline, denominator, latency = int(filters[options.denominator_correction]), fir_matrix = [denominator_correction[::-1]], time_domain = True)

	# Interleave the channels to make one stream
	channels = calibration_parts.mkinterleave(pipeline, [numerator, denominator])
	# Remove any possibly troublesome data from the beginning
	if chop_samples > 0:
		channels = pipeparts.mkgeneric(pipeline, channels, "lal_insertgap", insert_gap = False, chop_length = int(chop_samples * 1000000000 / sample_rate))
	# Send the data to lal_transferfunction to compute and write transfer functions
	pipeparts.mkgeneric(pipeline, channels, "lal_transferfunction", fft_length = fft_length, fft_overlap = fft_overlap, num_ffts = num_ffts, fir_length = td_tf_length, use_median = True if options.use_median else False, update_samples = 1e15, filename = '%s_%s_over_%s_%d-%d.txt' % (ifo, options.numerator_channel_name, options.denominator_channel_name, options.gps_start_time, data_duration))

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

# Remove unwanted lines from file, and re-format wanted lines
f = open('%s_%s_over_%s_%d-%d.txt' % (ifo, options.numerator_channel_name, options.denominator_channel_name, options.gps_start_time, data_duration),"r")
lines = f.readlines()
f.close()
f = open('%s_%s_over_%s_%d-%d.txt' % (ifo, options.numerator_channel_name, options.denominator_channel_name, options.gps_start_time, data_duration),"w")
for i in range(3, 3 + tf_length):
	f.write(lines[i].replace(' + ', '\t').replace(' - ', '\t-').replace('i', ''))
f.close()

# Read data from re-formatted file and find frequency vector, magnitude, and phase
data = numpy.loadtxt('%s_%s_over_%s_%d-%d.txt' % (ifo, options.numerator_channel_name, options.denominator_channel_name, options.gps_start_time, data_duration))
frequency = []
magnitude = []
phase = []
for i in range(0, len(data)):
	frequency.append(data[i][0])
	magnitude.append(abs(data[i][1] + 1j * data[i][2]))
	phase.append(numpy.angle(data[i][1] + 1j * data[i][2]) * 180.0 / numpy.pi)

# Make plots
freq_scale = 'log' if options.frequency_min > 0.0 and options.frequency_max / options.frequency_min > 10 else 'linear'
mag_scale = 'log' if options.magnitude_min > 0.0 and options.magnitude_max / options.magnitude_min > 10 else 'linear'
plt.figure(figsize = (10, 10))
ax = plt.subplot(211)
ax.set_xscale(freq_scale)
ax.set_yscale(mag_scale)
plt.plot(frequency, magnitude, label = '%s:%s / %s:%s' % (ifo, options.numerator_channel_name, ifo, options.denominator_channel_name))
plt.title(options.plot_title)
plt.ylabel('Magnitude')
plt.xlim(options.frequency_min, options.frequency_max)
plt.ylim(options.magnitude_min, options.magnitude_max)
plt.grid(True)
plt.legend()
ax = plt.subplot(212)
ax.set_xscale(freq_scale)
plt.plot(frequency, phase)
plt.ylabel('Phase [deg]')
plt.xlabel('Frequency [Hz]')
plt.xlim(options.frequency_min, options.frequency_max)
plt.ylim(options.phase_min, options.phase_max)
plt.grid(True)
plt.savefig('%s_%s_over_%s_%d-%d.png' % (ifo, options.numerator_channel_name, options.denominator_channel_name, options.gps_start_time, data_duration))


