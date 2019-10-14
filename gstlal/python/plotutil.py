# Copyright (C) 2013-2016 Kipp Cannon
# Copyright (C) 2015      Chad Hanna
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

##
# @file
#
# A file that contains some generic plotting module code
#
# ### Review Status
#
##
# @package python.plotutil
#
# plotting utilities module
#


import math
import numpy
import re
from matplotlib.colors import hex2color


golden_ratio = (1. + math.sqrt(5.)) / 2.


def colour_from_instruments(instruments, colours = {
	"G1": numpy.array(hex2color('#222222')),
	"H1": numpy.array(hex2color('#ee0000')),
	"L1": numpy.array(hex2color('#4ba6ff')),
	"V1": numpy.array(hex2color('#9b59b6')),
	"K1": numpy.array(hex2color('#ffb200')),
	"E1": numpy.array((1.0, 0.0, 0.0)),
	"E2": numpy.array((0.0, 0.8, 0.0)),
	"E3": numpy.array((1.0, 0.0, 1.0)),
}):
	# mix colours additively
	colour = sum(map(colours.__getitem__, instruments))
	# use single-instrument colours as-given
	if len(instruments) > 1:
		# desaturate
		colour += len(instruments) - 1
		# normalize
		colour /= colour.max()
	return colour


#
# =============================================================================
#
#                                 TeX Helpers
#
# =============================================================================
#


floatpattern = re.compile("([+-]?[.0-9]+)[Ee]([+-]?[0-9]+)")

def latexnumber(s):
	"""
	Convert a string of the form "d.dddde-dd" to "d.dddd \\times
	10^{-dd}".  Strings that contain neither an "e" nor an "E" are
	returned unchanged.
	"""
	if "e" not in s and "E" not in s:
		return s
	m, e = floatpattern.match(s).groups()
	return r"%s \times 10^{%d}" % (m, int(e))


def latexfilename(s):
	"""
	Escapes "\\" and "_" characters, and replaces " " with "~"
	(non-breaking space).
	"""
	return s.replace("\\", "\\\\").replace("_", "\\_").replace(" ", "~")
