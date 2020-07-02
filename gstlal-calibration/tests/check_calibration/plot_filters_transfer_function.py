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


import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy
from math import pi
import resource
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['mathtext.default'] = 'regular'
import glob
import matplotlib.pyplot as plt

from ticks_and_grid import ticks_and_grid

from optparse import OptionParser, Option

parser = OptionParser()
parser.add_option("--tf-file-directory", metavar = "directory", default = '.', help = "location of txt files with transfer functions (Default is current directory, '.')")
parser.add_option("--response-model-jump-delay", metavar = "seconds", type = float, default = 0.000061035, help = "Time delay in time-stamping DARM_ERR (Default is one 16384-Hz clock cycle)")
parser.add_option("--filters-model-jump-delay", metavar = "seconds", type = float, default = 0.0, help = "Any time delay in time-stamping DARM_ERR not contained in the model in the filters file (Default is 0.0 seconds).")
parser.add_option("--tf-frequency-min", type = float, default = -1, help = "Minimum frequency for transfer function plots (Default is to disable)")
parser.add_option("--tf-frequency-max", type = float, default = -1, help = "Maximum frequency for transfer function plots (Default is to disable)")
parser.add_option("--tf-frequency-scale", default = "log", help = "Frequency scale for transfer function plots (linear or log, default is log)")
parser.add_option("--tf-magnitude-min", type = float, default = -1, help = "Minimum magnitude for transfer function plots (Default is to disable)")
parser.add_option("--tf-magnitude-max", type = float, default = -1, help = "Maximum magnitude for transfer function plots (Default is to disable)")
parser.add_option("--tf-magnitude-scale", default = "log", help = "Magnitude scale for transfer function plots (linear or log, default is log)")
parser.add_option("--tf-phase-min", metavar = "degrees", type = float, default = 1000, help = "Minimum phase for transfer function plots, in degrees (Default is to disable)")
parser.add_option("--tf-phase-max", metavar = "degrees", type = float, default = 1000, help = "Maximum phase for transfer function plots, in degrees (Default is to disable)")
parser.add_option("--ratio-frequency-min", type = float, default = -1, help = "Minimum frequency for transfer function ratio plots (Default is to disable)")
parser.add_option("--ratio-frequency-max", type = float, default = -1, help = "Maximum frequency for transfer function ratio plots (Default is to disable)")
parser.add_option("--ratio-frequency-scale", default = "log", help = "Frequency scale for transfer function ratio plots (linear or log, default is log)")
parser.add_option("--ratio-magnitude-min", type = float, default = -1, help = "Minimum magnitude for transfer function ratio plots (Default is to disable)")
parser.add_option("--ratio-magnitude-max", type = float, default = -1, help = "Maximum magnitude for transfer function ratio plots (Default is to disable)")
parser.add_option("--ratio-magnitude-scale", default = "linear", help = "Magnitude scale for transfer function ratio plots (linear or log, default is linear)")
parser.add_option("--ratio-phase-min", metavar = "degrees", type = float, default = 1000, help = "Minimum phase for transfer function ratio plots, in degrees (Default is to disable)")
parser.add_option("--ratio-phase-max", metavar = "degrees", type = float, default = 1000, help = "Maximum phase for transfer function ratio plots, in degrees (Default is to disable)")

options, filenames = parser.parse_args()


# FIXME: Hard-coded CALCS dewhitening stuff.  We should have a file with a history of this for H1 and L1, or something like that.
zeros = [30+0j,30+0j,30+0j,30+0j,30+0j,30+0j,-3.009075115760242e3+3.993177550236464e3j,-3.009075115760242e3+-3.993177550236464e3j,-5.839434764093102e2+6.674504477214695e3j,-5.839434764093102e2-6.674504477214695e3j]
poles = [0.3+0j,0.3+0j,0.3+0j,0.3+0j,0.3+0j,0.3+0j,1.431097327857237e2+8.198751100282409e3j,1.431097327857237e2-8.198751100282409e3j,8.574723070843939e2+1.636154629741894e4j,8.574723070843939e2-1.636154629741894e4j]

#
# Load in the filters file that contains filter coefficients, etc.
# Search the directory tree for files with names matching the one we want.
#

# Identify any files that have filters transfer functions in them
tf_files = [f for f in os.listdir(options.tf_file_directory) if (os.path.isfile(f) and ('GDS' in f or 'DCS' in f) and 'npz' in f and 'filters_transfer_function' in f and '.txt' in f)]
# Check if we have something from CALCS data
calcs_tf_file = [f for f in os.listdir(options.tf_file_directory) if (os.path.isfile(f) and 'CALCS_response' in f and '.txt' in f)]
if len(calcs_tf_file):
	tf_files.append(calcs_tf_file[0])

# Move response function transfer function to the end so they can be plotted together
for i in range(0, len(tf_files)):
	if '_response_filters_transfer_function_' in tf_files[i] and "CALCS" in tf_files[i]:
		response_file = tf_files.pop(i)
		tf_files.append(response_file)
for i in range(0, len(tf_files)):
	if '_response_filters_transfer_function_' in tf_files[i] and "GDS" in tf_files[i]:
		response_file = tf_files.pop(i)
		tf_files.append(response_file)
for i in range(0, len(tf_files)):
	if '_response_filters_transfer_function_' in tf_files[i] and "DCS" in tf_files[i]:
		response_file = tf_files.pop(i)
		tf_files.append(response_file)

if '1262900044' in tf_files[-1]:
	response_file = tf_files.pop(len(tf_files) - 2)
	tf_files.append(response_file)

found_response = False
response_count = 0

# Plot limits
freq_min = options.tf_frequency_min if options.tf_frequency_min > 0 else None
freq_max = options.tf_frequency_max if options.tf_frequency_max > 0 else None
mag_min = options.tf_magnitude_min if options.tf_magnitude_min > 0 else None
mag_max = options.tf_magnitude_max if options.tf_magnitude_max > 0 else None
phase_min = options.tf_phase_min if options.tf_phase_min < 1000 else None
phase_max = options.tf_phase_max if options.tf_phase_max < 1000 else None
ratio_freq_min = options.ratio_frequency_min if options.ratio_frequency_min > 0 else None
ratio_freq_max = options.ratio_frequency_max if options.ratio_frequency_max > 0 else None
ratio_mag_min = options.ratio_magnitude_min if options.ratio_magnitude_min > 0 else None
ratio_mag_max = options.ratio_magnitude_max if options.ratio_magnitude_max > 0 else None
ratio_phase_min = options.ratio_phase_min if options.ratio_phase_min < 1000 else None
ratio_phase_max = options.ratio_phase_max if options.ratio_phase_max < 1000 else None

for tf_file in tf_files:
	filters_name = None
	if '_npz' in tf_file:
		filters_name = tf_file.split('_npz')[0] + '.npz'
	elif '_response_filters_transfer_function_' in tf_file:
		for tf_file_backup in tf_files:
			if '_npz' in tf_file_backup:
				filters_name = tf_file_backup.split('_npz')[0] + '.npz'
	if filters_name is not None:
		# Search the directory tree for filters files with names matching the one we want.
		filters_paths = []
		print("\nSearching for %s ..." % filters_name)
		# Check the user's home directory
		for dirpath, dirs, files in os.walk(os.environ['PWD'] + '/Filters'):
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
	else:
		filters = []

	model_jump_delay = options.filters_model_jump_delay
	# Determine what transfer function this is
	if '_tst_' in tf_file and 'DCS' in tf_file:
		plot_title = "TST Transfer Function"
		model_name = "tst_model" if "tst_model" in filters else None
	elif '_tst_' in tf_file and 'GDS' in tf_file:
		plot_title = "TST Correction Transfer Function"
		model_name = "tst_model" if "tst_model" in filters else "TST_corr_model" if "TST_corr_model" in filters else "ctrl_corr_model" if "ctrl_corr_model" in filters else None
	elif '_pum_' in tf_file and 'DCS' in tf_file:
		plot_title = "PUM Transfer Function"
		model_name = "pum_model" if "pum_model" in filters else None
	elif '_pum_' in tf_file and 'GDS' in tf_file:
		plot_title = "PUM Correction Transfer Function"
		model_name = "pum_model" if "pum_model" in filters else "PUM_corr_model" if "PUM_corr_model" in filters else "ctrl_corr_model" if "ctrl_corr_model" in filters else None
	elif '_uim_' in tf_file and 'DCS' in tf_file:
		plot_title = "UIM Transfer Function"
		model_name = "uim_model" if "uim_model" in filters else None
	elif '_uim_' in tf_file and 'GDS' in tf_file:
		plot_title = "UIM Correction Transfer Function"
		model_name = "uim_model" if "uim_model" in filters else "UIM_corr_model" if "UIM_corr_model" in filters else "ctrl_corr_model" if "ctrl_corr_model" in filters else None
	elif '_pumuim_' in tf_file and 'DCS' in tf_file:
		plot_title = "PUM/UIM Transfer Function"
		model_name = "pumuim_model" if "pumuim_model" in filters else None
	elif '_pumuim_' in tf_file and 'GDS' in tf_file:
		plot_title = "PUM/UIM Correction Transfer Function"
		model_name = "pumuim_model" if "pumuim_model" in filters else "ctrl_corr_model" if "ctrl_corr_model" in filters else None
	elif '_res_' in tf_file and 'DCS' in tf_file:
		plot_title = "Inverse Sensing Transfer Function"
		model_name = "invsens_model" if "invsens_model" in filters else None
	elif '_res_' in tf_file and 'GDS' in tf_file:
		plot_title = "Inverse Sensing Correction Transfer Function"
		model_name = "invsens_model" if "invsens_model" in filters else "res_corr_model" if "res_corr_model" in filters else None
	elif '_response_filters_transfer_function_' in tf_file:
		found_response = True
		plot_title = "Response Function"
		model_name = "response_function" if "response_function" in filters else None
		model_jump_delay = options.response_model_jump_delay
	else:
		plot_title = "Transfer Function"
		model_name = None

	ifo = ''
	if 'H1' in tf_file:
		ifo = 'H1'
	if 'L1' in tf_file:
		ifo = 'L1'

	if 'CALCS' in tf_file:
		cal_version = 'Front\\mbox{-}end'
		color = 'silver'
	elif 'GDS' in tf_file:
		if '1262900044' in tf_file:
			cal_version = 'GDS'
			color = 'royalblue'
		else:
			cal_version = 'MoreTSTCorrections'
			color = 'maroon'
	elif 'DCS' in tf_file:
		cal_version = 'DCS'
		color = 'maroon'
	else:
		cal_version = 'Filters'

	if '_tst_' in tf_file:
		component = 'TST'
	elif '_pum_' in tf_file:
		component = 'PUM'
	elif '_uim_' in tf_file:
		component = 'UIM'
	elif '_pumuim_' in tf_file:
		component = 'PUMUIM'
	elif '_tst_' in tf_file:
		component = 'TST'
	elif '_res_' in tf_file:
		component = 'C inv'
	elif '_response_filters_transfer_function_' in tf_file:
		component = 'Response'

	# Remove unwanted lines from transfer function file, and re-format wanted lines
	f = open(tf_file, 'r')
	lines = f.readlines()
	f.close()
	tf_length = len(lines) - 5
	f = open(tf_file.replace('.txt', '_reformatted.txt'), 'w')
	for j in range(3, 3 + tf_length):
		f.write(lines[j].replace(' + ', '\t').replace(' - ', '\t-').replace('i', ''))
	f.close()

	# Read data from re-formatted file and find frequency vector, magnitude, and phase
	data = numpy.loadtxt(tf_file.replace('.txt', '_reformatted.txt'))
	os.remove(tf_file.replace('.txt', '_reformatted.txt'))
	filters_tf = []
	frequency = []
	magnitude = []
	phase = []
	df = data[1][0] - data[0][0] # frequency spacing
	for j in range(0, len(data)):
		frequency.append(data[j][0])
		tf_at_f = (data[j][1] + 1j * data[j][2]) * numpy.exp(-2j * numpy.pi * data[j][0] * model_jump_delay)
		if 'CALCS' in tf_file:
			# Apply dewhitening here
			for z in zeros:
				tf_at_f = tf_at_f * (1.0 + 1j * frequency[j] / z)
			for p in poles:
				tf_at_f = tf_at_f / (1.0 + 1j * frequency[j] / p)
		filters_tf.append(tf_at_f)
		magnitude.append(abs(tf_at_f))
		phase.append(numpy.angle(tf_at_f) * 180.0 / numpy.pi)

	# Find frequency-domain models in filters file if they are present and resample if necessary
	if model_name is not None:
		model = filters[model_name]
		model_tf = []
		model_magnitude = []
		model_phase = []
		ratio = []
		ratio_magnitude = []
		ratio_phase = []
		# Check the frequency spacing of the model
		model_df = model[0][1] - model[0][0]
		cadence = df / model_df
		index = 0
		# This is a linear resampler (it just connects the dots with straight lines)
		while index < tf_length:
			before_idx = numpy.floor(cadence * index)
			after_idx = numpy.ceil(cadence * index + 1e-10)
			# Check if we've hit the end of the model transfer function
			if after_idx >= len(model[0]):
				if before_idx == cadence * index:
					model_tf.append(model[1][before_idx] + 1j * model[2][before_idx])
					model_magnitude.append(abs(model_tf[index]))
					model_phase.append(numpy.angle(model_tf[index]) * 180.0 / numpy.pi)
					ratio.append(filters_tf[index] / model_tf[index])
					ratio_magnitude.append(abs(ratio[index]))
					ratio_phase.append(numpy.angle(ratio[index]) * 180.0 / numpy.pi)
				index = tf_length
			else:
				before = model[1][before_idx] + 1j * model[2][before_idx]
				after = model[1][after_idx] + 1j * model[2][after_idx]
				before_weight = after_idx - cadence * index
				after_weight = cadence * index - before_idx
				model_tf.append(before_weight * before + after_weight * after)
				model_magnitude.append(abs(model_tf[index]))
				model_phase.append(numpy.angle(model_tf[index]) * 180.0 / numpy.pi)
				ratio.append(filters_tf[index] / model_tf[index])
				ratio_magnitude.append(abs(ratio[index]))
				ratio_phase.append(numpy.angle(ratio[index]) * 180.0 / numpy.pi)
				index += 1

		numpy.savetxt(tf_file.replace('.txt', '_ratio_magnitude.txt'), numpy.transpose(numpy.array([frequency, ratio_magnitude])), fmt='%.5e', delimiter='\t')
		numpy.savetxt(tf_file.replace('.txt', '_ratio_phase.txt'), numpy.transpose(numpy.array([frequency, ratio_phase])), fmt='%.5f', delimiter='\t')

	# Filter transfer function plots
	if not response_count:
		plt.figure(figsize = (10, 8))
	if model_name is not None and model_name is not 'response_function':
		plt.subplot(221)
		plt.plot(frequency, model_magnitude, 'orangered', linewidth = 1.0, label = r'${\rm %s \ Model \ %s}$' % (ifo, component))
		leg = plt.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		#plt.title(plot_title)
		plt.ylabel(r'${\rm Magnitude \ [m/ct]}$')
		ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = mag_min, ymax = mag_max, xscale = options.tf_frequency_scale, yscale = options.tf_magnitude_scale)
		ax = plt.subplot(223)
		plt.plot(frequency, model_phase, 'orangered', linewidth = 1.0)
		plt.ylabel(r'${\rm Phase \ [deg]}$')
		plt.xlabel(r'${\rm Frequency \ [Hz]}$')
		ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = phase_min, ymax = phase_max, xscale = options.tf_frequency_scale)
	plt.subplot(221)
	plt.plot(frequency, magnitude, color, linewidth = 1.0, label = r'${\rm %s \ %s \ %s}$' % (ifo, cal_version, component))
	leg = plt.legend(fancybox = True)
	leg.get_frame().set_alpha(0.5)
	plt.ylabel(r'${\rm Magnitude \ [m/ct]}$')
	ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = mag_min, ymax = mag_max, xscale = options.tf_frequency_scale, yscale = options.tf_magnitude_scale)
	ax = plt.subplot(223)
	plt.plot(frequency, phase, color, linewidth = 1.0)
	plt.ylabel(r'${\rm Phase [deg]}$')
	plt.xlabel(r'${\rm Frequency \ [Hz]}$')
	ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = phase_min, ymax = phase_max, xscale = options.tf_frequency_scale)

	# Plots of the ratio filters / model
	if model_name is not None:
		#plt.figure(figsize = (10, 12))
		plt.subplot(222)
		plt.plot(frequency, ratio_magnitude, color, linewidth = 1.0, label = r'${\rm %s \ %s / Model}$' % (ifo, cal_version))
		leg = plt.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		#plt.title(plot_title)
		#plt.ylabel(r'${\rm Magnitude \ [m/ct]}$')
		ticks_and_grid(plt.gca(), xmin = ratio_freq_min, xmax = ratio_freq_max, ymin = ratio_mag_min, ymax = ratio_mag_max, xscale = options.ratio_frequency_scale, yscale = options.ratio_magnitude_scale)
		ax = plt.subplot(224)
		plt.plot(frequency, ratio_phase, color, linewidth = 1.0)
		#plt.ylabel(r'${\rm Phase \ [deg]}$')
		plt.xlabel(r'${\rm Frequency \ [Hz]}$')
		ticks_and_grid(plt.gca(), xmin = ratio_freq_min, xmax = ratio_freq_max, ymin = ratio_phase_min, ymax = ratio_phase_max, xscale = options.ratio_frequency_scale)
		if not ('_response_filters_transfer_function_' in tf_file):
			plt.savefig(tf_file.replace('.txt', '_ratio.png'))
			plt.savefig(tf_file.replace('.txt', '_ratio.pdf'))
	if '_response_filters_transfer_function_' in tf_file:
		response_count += 1

if response_count:
	# Now add the model response function
	plt.subplot(221)
	plt.plot(frequency, model_magnitude, 'orangered', linewidth = 1.0, ls = '--', label = r'${\rm %s \ Model \ %s}$' % (ifo, component))
	leg = plt.legend(fancybox = True)
	leg.get_frame().set_alpha(0.5)
	#plt.title(plot_title)
	plt.ylabel(r'${\rm Magnitude \ [m/ct]}$')
	ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = mag_min, ymax = mag_max, xscale = options.tf_frequency_scale, yscale = options.tf_magnitude_scale)
	ax = plt.subplot(223)
	plt.plot(frequency, model_phase, 'orangered', linewidth = 1.0, ls = '--')
	plt.ylabel(r'${\rm Phase \ [deg]}$')
	plt.xlabel(r'${\rm Frequency \ [Hz]}$')
	ticks_and_grid(plt.gca(), xmin = freq_min, xmax = freq_max, ymin = phase_min, ymax = phase_max, xscale = options.tf_frequency_scale)
plt.savefig(tf_file.replace('.txt', '_ratio.png'))
plt.savefig(tf_file.replace('.txt', '_ratio.pdf'))



