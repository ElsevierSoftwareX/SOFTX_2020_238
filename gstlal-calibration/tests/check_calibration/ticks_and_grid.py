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


import matplotlib as mpl; mpl.use('Agg')

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

def find_minor_ticks(major_ticks, scale = 'linear'):

	major_ticks = np.copy(major_ticks)

	if len(major_ticks) < 2:
		return np.array([])
	elif scale == 'linear':
		major_spacing = major_ticks[1] - major_ticks[0]
		major_spacing_int = int(round(major_spacing / pow(10, int(np.log10(major_spacing)) - 2)))

		if major_spacing_int >= 30:
			major_spacing_int = major_spacing_int // 10

		if major_spacing_int % 5 == 0:
			minor_spacing = major_spacing / 5
			cadence = 4
		elif major_spacing_int % 4 == 0:
			minor_spacing = major_spacing / 4
			cadence = 3
		elif major_spacing_int % 3 == 0:
			minor_spacing = major_spacing / 3
			cadence = 2
		else:
			minor_spacing = major_spacing / 5
			cadence = 4

		minor_ticks = np.zeros((len(major_ticks) - 1) * cadence)
		for i in range(cadence):
			minor_ticks[i::cadence] = major_ticks[:-1] + (i + 1) * minor_spacing
		return minor_ticks

		return np.arange(major_ticks[0], major_ticks[-1] + float(minor_spacing) / 2, minor_spacing)
	else:
		ratio = int(round(float(major_ticks[1]) / major_ticks[0]))
		if ratio == 10:
			cadence = 8 if len(major_ticks) < 7 else 4
			factors = np.array([2,3,4,5,6,7,8,9]) if cadence == 8 else np.array([2,4,6,8])
			minor_ticks = np.zeros((len(major_ticks) - 1) * cadence)
			for i in range(cadence):
				minor_ticks[i::cadence] = factors[i] * major_ticks[:-1]
			return minor_ticks
		elif ratio >= 100:
			cadence = int(np.log10(ratio) - 1)
			minor_ticks = np.zeros((len(major_ticks) - 1) * cadence)
			for i in range(cadence):
				minor_ticks[i::cadence] = pow(10, i + 1) * major_ticks[:-1]
			return minor_ticks
		else:
			return np.array([])

def ticks_and_grid(ax, xmin = None, xmax = None, ymin = None, ymax = None, xscale = None, yscale = None):

	xlim = ax.get_xlim()
	xmin = xlim[0] if xmin is None else xmin
	xmax = xlim[1] if xmax is None else xmax
	ylim = ax.get_ylim()
	ymin = ylim[0] if ymin is None else ymin
	ymax = ylim[1] if ymax is None else ymax

	if xscale is None:
		xscale = ax.get_xscale()
	if yscale is None:
		yscale = ax.get_yscale()

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_xscale(xscale)
	ax.set_yscale(yscale)

	major_xticks = ax.get_xticks()
	major_yticks = ax.get_yticks()

	ax.set_yticks(find_minor_ticks(major_yticks, scale = yscale), minor = True)
	ax.set_xticks(find_minor_ticks(major_xticks, scale = xscale), minor = True)
	ax.set_yticks(major_yticks)
	ax.set_xticks(major_xticks)
	ax.set_axisbelow(True)
	ax.grid(True, which = "major", ls = '-', linewidth = 0.5, color = 'dimgray')
	ax.grid(True, which = "minor", ls = '-', linewidth = 0.5, color = 'lightgray')

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)

	ax.set_xticklabels([], minor = True)
	ax.set_yticklabels([], minor = True)


