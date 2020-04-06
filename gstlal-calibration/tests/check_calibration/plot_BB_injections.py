#!/usr/bin/env python
# Copyright (C) 2020  Aaron Viets
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


import sys
import os
import subprocess
import time
import numpy as np
import xml.etree.ElementTree as ET
from optparse import OptionParser, Option

parser = OptionParser()
parser.add_option("--ifo", metavar = "name", type = str, default = "H", help = "Name of the interferometer (IFO), e.g., H, L.  Default is H.")
parser.add_option("--obs-run", metavar = "name", type = str, default = "O3", help = "Name of the observing run, e.g., O1, O2, O3.  Default is O3.")
parser.add_option("--xml-filename", metavar = "filename", type = str, default = None, help = "Path to xml file with information about the Pcal broadband injection")
parser.add_option("--check-directory", metavar = "directory", type = str, default = None, help = "Path to a directory with xml files.  The code will check once each day if there is a new file, and if/when there is (and there is h(t) data available), it will automatically make a plot.")
parser.add_option("--plots-directory", metavar = "directory", type = str, default = None, help = "Path to a directory in which to put plots")
parser.add_option("--update-svn", action = "store_true", help = "If set, the check-directory and plots-directory will be updated each time they are \"checked,\" and new files in the plots-directory will be committed to the calibration svn.")
parser.add_option("--RX-or-TX", type = str, default = 'RX', help = "Which Pcal channel to use: RX or TX.  Default is RX.")
parser.add_option("--fmin", metavar = "Hz", type = float, default = 10.0, help = "minimum frequency for plot")
parser.add_option("--fmax", metavar = "Hz", type = float, default = 400.0, help = "maximum frequency for plot")
parser.add_option("--magnitude-min", type = float, default = 0.9, help = "minimum magnitude for plot")
parser.add_option("--magnitude-max", type = float, default = 1.1, help = "maximum magnitude for plot")
parser.add_option("--phase-min", type = float, default = -6.0, help = "minimum phase for plot")
parser.add_option("--phase-max", type = float, default = 6.0, help = "maximum phase for plot")
parser.add_option("--BB-duration", type = int, metavar = "seconds", default = 180, help = "Duration in seconds of the Pcal broadband injection.")
parser.add_option("--check-period", type = int, metavar = "seconds", default = 86400, help = "How long in seconds to wait between times when to check the check-directory for new xml files (Default is one day)")
parser.add_option("--cal-versions", type = str, default = 'R,HOFT_C00', help = "Comma-separated list of versions of calibration to plot. Versions include R, HOFT_C00, HOFT_C01, and HOFT_C02. Plot will not be produced until all versions are available. (Default is \'R,HOFT_C00\')")
parser.add_option("--filters-file", type = str, default = None, help = "Name of the filters file used to calibrate the data")

options, filenames = parser.parse_args()

if options.xml_filename is None and options.check_directory is None:
	raise ValueError("Either --xml-filename or --check-directory must be set for this script to do anything.")

cal_versions = options.cal_versions.split(',')
# Make strings with comma-separated lists of corresponding h(t) channels and labels for plots
hoft_channel_list = ''
labels = ''
cal_scale_factors = ''
zeros = ''
poles = ''
for version in cal_versions:
	if version == 'R':
		hoft_channel_list += 'CAL-DELTAL_EXTERNAL_DQ,'
		labels += 'CALCS,'
		cal_scale_factors += 'None,'
		zeros += '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3;'
		poles += '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4;'
	elif version == 'HOFT_C00':
		hoft_channel_list += 'GDS-CALIB_STRAIN,'
		labels += 'C00,'
		cal_scale_factors += 'arm_length,'
		zeros += ';'
		poles += ';'
	elif version == 'HOFT_C01':
		hoft_channel_list += 'DCS-CALIB_STRAIN_C01,'
		labels += 'C01,'
		cal_scale_factors += 'arm_length,'
		zeros += ';'
		poles += ';'
	elif version == 'HOFT_C02':
		hoft_channel_list += 'DCS-CALIB_STRAIN_C02,'
		labels += 'C02,'
		cal_scale_factors += 'arm_length,'
		zeros += ';'
		poles += ';'
	else:
		raise ValueError("cal-versions must be a comma-separated list formatted as a string with no spaces.  The items in the list must be R, HOFT_C00, HOFT_C01, or HOFT_C02.")
hoft_channel_list = hoft_channel_list[:-1]
labels = labels[:-1]
cal_scale_factors = cal_scale_factors[:-1]
zeros = zeros[:-1]
poles = poles[:-1]

# Get a GPS start time from an xml file
def get_gps_start_time(xml_filename):
	tree = ET.parse(xml_filename)
	root = tree.getroot()

	# In case the format of the xml file is not always the same,
	# use some criteria to determine the best candidates for the
	# actual GPS start time.
	best = []
	better = []
	worst = []
	for element in root.find('LIGO_LW').findall('Time'):
		if element.get('Name') == 'TestTime' and element.get('Type') == 'GPS':
			try:
				best.append(int(float(element.text)))
			except:
				pass
		elif element.get('Name') == 'TestTime':
			try:
				better.append(int(float(element.text)))
			except:
				pass
		else:
			try:
				worst.append[int(float(element.text))]
			except:
				pass
	if any(best):
		return best[0]
	elif any(better):
		return better[0]
	elif any(worst):
		return worst[0]
	else:
		return 0


def get_pcal_channel(xml_filename, RX_or_TX):
	PCALY_or_PCALX = None
	if 'PCALY' in xml_filename:
		PCALY_or_PCALX = 'PCALY'
	elif 'PCALX' in xml_filename:
		PCALY_or_PCALX = 'PCALX'
	else:
		# grep the file and see which string appears more times
		num_PCALY = len(os.popen('grep "PCALY" %s' % xml_filename).read().split('\n')) - 1
		num_PCALX = len(os.popen('grep "PCALX" %s' % xml_filename).read().split('\n')) - 1
		if num_PCALY <= 0 and num_PCALX <= 0:
			PCALY_or_PCALX = None
		elif num_PCALY >= num_PCALX:
			PCALY_or_PCALX = 'PCALY'
		else:
			PCALY_or_PCALX = 'PCALX'

	if PCALY_or_PCALX == None:
		return None
	else:
		return 'CAL-%s_%s_PD_OUT_DQ' % (PCALY_or_PCALX, RX_or_TX)


def get_frame_cache(ifo, frame_type, gps_start_time, gps_end_time):
	# Use gw_data_find to make a frame cache file with a unique name
	cache_name = '%s1_%s_BBframes_%d-%d.cache' % (ifo, frame_type, gps_start_time, gps_end_time - gps_start_time)
	os.system('gw_data_find -o %s -t %s1_%s -s %d -e %d -l --url-type file > %s' % (ifo, ifo, frame_type, gps_start_time, gps_end_time, cache_name))
	# Check to see if data is there for the entire time interval
	cache_file = open(cache_name, 'r')
	lines = cache_file.readlines()
	cache_file.close()
	if not any(lines):
		return 'None'
	cache_start = int(lines[0].split(' ')[2])
	cache_end = int(lines[-1].split(' ')[2]) + int(lines[-1].split(' ')[3])
	if cache_start <= gps_start_time and cache_end >= gps_end_time:
		return cache_name
	else:
		return 'None'


def find_filters_file(ifo, gps_start_time, update_svn, obs_run):
	if update_svn:
		try:
			os.system('svn up Filters/%s/GDSFilters' % obs_run)
		except:
			pass

	# Find the filters file most likely to be correct.
	GDSFilters_files = os.popen('ls Filters/%s/GDSFilters -p | grep -v /' % obs_run).read().split('\n')
	all_filters_files = []
	for f in GDSFilters_files:
		if '%s1' % ifo in f and '.npz' in f:
			all_filters_files.append(f)

	before_filt_times = []
	after_filt_times = []
	for f in all_filters_files:
		filt_time = ''
		for char in f:
			if char.isdigit():
				filt_time += char
			else:
				# An appropriate GPS time should have 10 digits. and be
				if len(filt_time) == 10:
					# Ideally, pick one from before the gps_start_time of
					# the Pcal broadband injection.
					if int(filt_time) < gps_start_time:
						before_filt_times.append(int(filt_time))
					else:
						after_filt_times.append(int(filt_time))
				filt_time = ''

	# Choose the time closest to the gps_start_time
	if any(before_filt_times):
		chosen_filt_time = max(before_filt_times)
	else:
		# If there aren't any filters from before the gps_start_time, find one after.
		chosen_filt_time = min(after_filt_times)

	best_filter_files = []
	for f in all_filters_files:
		if str(chosen_filt_time) in f:
			best_filter_files.append(f)

	# The shortest filename is a good bet.
	lengths = [len(ele) for ele in best_filter_files]
	length, idx = min((val, idx) for (idx, val) in enumerate(lengths))

	return best_filter_files[idx]


def make_plot(options, path_to_xml, path_to_plot, cal_versions, hoft_channel_list, labels, cal_scale_factors):
	ifo = options.ifo
	# Parse xml file to find GPS start time.
	gps_start_time = get_gps_start_time(path_to_xml) + 1
	gps_end_time = gps_start_time + options.BB_duration - 2

	# Get name of Pcal channel to use.
	pcal_channel_name = get_pcal_channel(path_to_xml, options.RX_or_TX)

	# Find the data we need, starting with raw frames with Pcal.
	pcal_frame_cache = hoft_frame_cache_list = 'None'
	if gps_start_time > 0:
		pcal_frame_cache = get_frame_cache(ifo, 'R', gps_start_time, gps_end_time)
	# Comma-separated list of calibrated frame caches
	if gps_start_time > 0:
		hoft_frame_cache_list = get_frame_cache(ifo, cal_versions[0], gps_start_time, gps_end_time)
		for version in cal_versions[1:]:
			hoft_frame_cache_list += ','
			hoft_frame_cache_list += get_frame_cache(ifo, version, gps_start_time, gps_end_time)
	# Find the best filters file for this data
	if gps_start_time > 0:
		filters_file = options.filters_file if options.filters_file is not None else find_filters_file(ifo, gps_start_time, options.update_svn, options.obs_run)

	if not ('None' in pcal_frame_cache or 'None' in hoft_frame_cache_list):
		# Then the data exists and we can make plots.
		os.system('python plot_transfer_function.py --gps-start-time %d --gps-end-time %d --ifo %s1 --denominator-frame-cache %s --denominator-channel-name %s --denominator-correction y_arm_pcal_corr --numerator-correction %s --zeros \'%s\' --poles \'%s\' --frequency-min %f --frequency-max %f --magnitude-min %f --magnitude-max %f --phase-min %f --phase-max %f --numerator-frame-cache-list %s --numerator-channel-list %s --filters-file %s --use-median --labels %s --filename %s' % (gps_start_time, gps_end_time, ifo, pcal_frame_cache, pcal_channel_name, cal_scale_factors, zeros, poles, float(options.fmin), float(options.fmax), float(options.magnitude_min), float(options.magnitude_max), float(options.phase_min), float(options.phase_max), hoft_frame_cache_list, hoft_channel_list, filters_file, labels, path_to_plot))
		return True
	else:
		return False


if options.xml_filename is not None:
	plots_directory = '.' if options.plots_directory is None else options.plots_directory
	path_to_plot = os.path.join(plots_directory, options.xml_filename).replace('.xml', '_%s' % labels.replace(',', '_'))

	# Update directories in the svn if requested.
	if options.update_svn:
		if options.check_directory is not None:
			try:
				os.system('svn up %s' % check_directory)
			except:
				pass
		if options.plots_directory is not None:
			try:
				os.system('svn up %s' % plots_directory)
			except:
				pass

	success = make_plot(options, options.xml_filename, path_to_plot, cal_versions, hoft_channel_list, labels, cal_scale_factors)

	if success and options.update_svn and options.plots_directory is not None:
		# Keep trying to update the svn for half of the check_period
		try_time = 0
		while try_time < options.check_period / 2:
			try:
				os.system('svn add %s.*' % path_to_plot)
				os.system('svn ci %s -m \"Plots of %s1 Pcal broadband injections\"' % (plots_directory, options.ifo))
				break
			except:
				# Wait 5 minutes and try again
				time.sleep(300)
				try_time += 300


if options.check_directory is not None:
	check_directory = options.check_directory
	if options.plots_directory is None:
		raise ValueError("--plots-directory must be set if --check-directory is set.")
	else:
		plots_directory = options.plots_directory

	while(True):
		# Update directories in the svn if requested.
		if options.update_svn:
			try:
				os.system('svn up %s' % check_directory)
				os.system('svn up %s' % plots_directory)
			except:
				pass

		# Check if there is a file in the check_directory
		# that has no counterpart in the plots directory.
		check_files = os.popen('ls -R %s -p | grep -v /' % check_directory).read().split('\n')
		# Which of these files should have a corresponding plot?
		BBinj_files = []
		for f in check_files:
			if '.xml' in f and 'BB' in f and 'PCAL' in f:
				BBinj_files.append(f)
		# Names of plot files that either exist or should be produced
		needed_plots = []
		for f in BBinj_files:
			needed_plots.append(f.replace('.xml', '_%s.png' % labels.replace(',', '_')))
		plot_files = os.popen('ls -R %s -p | grep -v /' % plots_directory).read().split('\n')

		for f in needed_plots:
			if not (f in plot_files):
				# Then we need to make a plot
				# Find location of broadband injection file
				path_to_BBinj_file = None
				for path, dirs, files in os.walk(check_directory):
					if f.replace('_%s.png' % labels.replace(',', '_'), '.xml') in files:
						path_to_BBinj_file = os.path.join(path, f.replace('_%s.png' % labels.replace(',', '_'), '.xml'))

				path_to_plot = os.path.join(plots_directory, f).replace('.png', '')

				success = make_plot(options, path_to_BBinj_file, path_to_plot, cal_versions, hoft_channel_list, labels, cal_scale_factors)

				if success and options.update_svn:
					try:
						os.system('svn add %s.*' % path_to_plot)
						os.system('svn ci %s -m \"Plots of %s1 Pcal broadband injections\"' % (plots_directory, options.ifo))
					except:
						pass

		if '?       ' in os.popen('svn status %s' % plots_directory).read():
			# Then there are unadded files that we should add.
			try:
				os.system('svn add %s/*' % plots_directory)
			except:
				pass
		if 'A       ' in os.popen('svn status %s' % plots_directory).read():
			# Then there are added files that we should commit.
			try:
				os.system('svn ci %s -m \"Plots of %s1 Pcal broadband injections\"' % (plots_directory, options.ifo))
			except:
				pass

		time.sleep(options.check_period)


