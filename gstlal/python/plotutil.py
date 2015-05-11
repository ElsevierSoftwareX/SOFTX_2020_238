# Copyright (C) 2013-2015 Kipp Cannon
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
import matplotlib
matplotlib.rcParams.update({
	"font.size": 10.0,
	"axes.titlesize": 10.0,
	"axes.labelsize": 10.0,
	"xtick.labelsize": 8.0,
	"ytick.labelsize": 8.0,
	"legend.fontsize": 8.0,
	"figure.dpi": 600,
	"savefig.dpi": 600,
	"text.usetex": True
})
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy

def colour_from_instruments(instruments, colours = {
	"G1": numpy.array((0.0, 1.0, 1.0)),
	"H1": numpy.array((1.0, 0.0, 0.0)),
	"H2": numpy.array((0.0, 0.0, 1.0)),
	"L1": numpy.array((0.0, 0.8, 0.0)),
	"V1": numpy.array((1.0, 0.0, 1.0)),
	"E1": numpy.array((1.0, 0.0, 0.0)),
	"E2": numpy.array((0.0, 0.8, 0.0)),
	"E3": numpy.array((1.0, 0.0, 1.0)),
}):
	# mix colours additively
	colour = sum(map(colours.__getitem__, instruments))
	# desaturate
	colour += len(instruments) - 1
	# normalize
	return colour / colour.max()

golden_ratio = (1. + math.sqrt(5.)) / 2. 

