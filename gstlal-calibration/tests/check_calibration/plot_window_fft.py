#!/usr/bin/env python3
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


import numpy as np

import matplotlib
from ticks_and_grid import ticks_and_grid
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from gstlal import FIRtools as fir


#
# A function to plot the frequency-response of a window function
#


def plot_window_fft(windows, sr, labels, filename = "window_fft.png"):
	if type(windows) is not list:
		windows = [windows]
	if type(labels) is not list:
		labels = [labels]
	for i in range(len(windows) - len(labels)):
		labels.append(labels[-1])

	times = []
	freqs = []
	freqresps = []
	fmax_zoom = 0.0
	magmin_zoom = 1.0
	for i in range(len(windows)):
		# Time vectors
		dur = float(len(windows[i])) / sr
		t = np.arange(0, dur, 1.0 / sr)
		t = t - t[-1] / 2.0
		times.append(t)

		# Frequency vectors
		freq = np.fft.rfftfreq(8 * len(t), d = 1.0 / sr)
		for j in range(1, len(freq)):
			freq = np.insert(freq, 0, -freq[2 * j - 1])
		freqs.append(freq)

		# Magnitudes of frequency responses of each window
		fresp = abs(fir.freqresp(windows[i]))
		fresp /= max(fresp)
		for j in range(1, len(fresp)):
			fresp = np.insert(fresp, 0, fresp[2 * j - 1])
		freqresps.append(fresp)

		# Find the width of the main lobe to decide plot limits.
		df = 1.0 / dur / 8.0
		fmax = 0.0
		magmin = 1.0
		for j in range(len(fresp) // 2, len(fresp) - 1):
			if fresp[j + 1] > fresp[j]:
				fmax = 5 * df * (j - len(fresp) // 2)
				magmin = np.percentile(fresp[len(fresp) // 2 : len(fresp) // 2 + 5 * (j - len(fresp) // 2)], 1)
				break
		fmax_zoom = max(fmax_zoom, fmax)
		magmin_zoom = min(magmin_zoom, magmin)

	# Now make the figure with three plots:
	colors = ['royalblue', 'maroon', 'springgreen', 'red', 'gold', 'magenta', 'orange', 'aqua', 'darkgreen', 'blue']
	plt.figure(figsize = (12, 4))
	plt.gcf().subplots_adjust(bottom=0.25)
	ax = plt.subplot(131)
	for i in range(len(windows)):
		plt.plot(times[i], windows[i], colors[i % 10], linewidth = 0.75)
	plt.ylabel('Magnitude')
	plt.xlabel('Time (s)')
	ticks_and_grid(ax, xscale = 'linear', yscale = 'linear')

	ax = plt.subplot(132)
	for i in range(len(windows)):
		plt.plot(freqs[i], freqresps[i], colors[i % 10], linewidth = 0.75, label = labels[i])
	leg = plt.legend(fancybox = True, loc = 'upper right')
	leg.get_frame().set_alpha(1.0)
	plt.xlabel('Frequency (Hz)')
	ticks_and_grid(ax, xscale = 'linear', yscale = 'log')

	ax = plt.subplot(133)
	for i in range(len(windows)):
		plt.plot(freqs[i], freqresps[i], colors[i % 10], linewidth = 0.75)
	plt.xlabel('Frequency (Hz)')
	ticks_and_grid(ax, xmin = -fmax_zoom, xmax = fmax_zoom, ymin = magmin_zoom, xscale = 'linear', yscale = 'log')

	plt.tight_layout()

	plt.savefig(filename)



