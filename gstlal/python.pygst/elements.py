# Copyright (C) 2009--2011  LIGO Scientific Collaboration
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

## @file

## @package elements

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# GstCaps description of matplotlib's agg backend's pixel format
#


matplotlibcaps = (
	"video/x-raw-rgb, " +
	"bpp = (int) 32, " +
	"depth = (int) 32, " +
	"red_mask = (int) -16777216, " +
	"green_mask = (int) 16711680, " +
	"blue_mask = (int) 65280, " +
	"alpha_mask = (int) 255, " +
	"endianness = (int) 4321"
)
