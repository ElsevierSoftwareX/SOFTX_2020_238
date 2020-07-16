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

def find_minor_ticks(major_ticks, scale = 'linear'):

	major_ticks = np.copy(major_ticks)

	if len(major_ticks) < 2:
		return major_ticks
	elif scale == 'linear':
		major_spacing = major_ticks[1] - major_ticks[0]
		major_spacing_int = int(round(major_spacing / pow(10, int(np.log10(major_spacing)) - 2)))

		if major_spacing_int >= 30:
			major_spacing_int = major_spacing_int // 10

		if major_spacing_int % 5 == 0:
			minor_spacing = major_spacing / 5
		elif major_spacing_int % 4 == 0:
			minor_spacing = major_spacing / 4
		elif major_spacing_int % 3 == 0:
			minor_spacing = major_spacing / 3
		else:
			minor_spacing = major_spacing / 5

		return np.arange(major_ticks[0], major_ticks[-1] + float(minor_spacing) / 2, minor_spacing)
	else:
		ratio = int(round(float(major_ticks[1]) / major_ticks[0]))
		if ratio == 10:
			cadence = 9 if len(major_ticks) < 7 else 5
			factors = np.array([1,2,3,4,5,6,7,8,9]) if cadence == 9 else np.array([0,2,4,6,8])
			minor_ticks = np.zeros((len(major_ticks) - 1) * cadence + 1)
			minor_ticks[::cadence] = major_ticks
			for i in range(1, cadence):
				minor_ticks[i::cadence] = factors[i] * major_ticks[:-1]
			return minor_ticks
		elif ratio >= 100:
			cadence = int(np.log10(ratio))
			minor_ticks = np.zeros((len(major_ticks) - 1) * cadence + 1)
			minor_ticks[::cadence] = major_ticks
			for i in range(1, cadence):
				minor_ticks[i::cadence] = pow(10, i) * major_ticks[:-1]
			return minor_ticks
		else:
			return major_ticks


