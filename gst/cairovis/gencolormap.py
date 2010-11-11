# Copyright (C) 2010  Leo Singer
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
"""
Auto-generate colormap_data.c, which packages Matplotlib's color data.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from matplotlib.cm import datad
from inspect import isfunction
import sys

print """
/*
 * Copyright (c) 2010  Leo Singer
 *
 * Colormap data from Matplotlib's matplotlib.cm module, which is
 * Copyright (c) 2002-2009 John D. Hunter; All Rights Reserved
 *
 */

#include "colormap.h"
#include <gsl/gsl_interp.h>
#include <string.h>
gboolean colormap_get_data_by_name(gchar *name, colormap_data *data) {
"""

for key, value in sorted(datad.items()):
	if hasattr(value, 'iteritems') and not(isfunction(value['red']) or isfunction(value['green']) or isfunction(value['blue'])):
		print 'if (strcmp(name, "%s") == 0) {' % key
		for color in ('red', 'green', 'blue'):
			print '{'
			print 'const double x[] = {', ','.join([repr(x) for x, y0, y1 in sorted(value[color])]), '};'
			print 'const double y[] = {', ','.join([repr(y1) for x, y0, y1 in sorted(value[color])]), '};'
			print 'data->%s.len = sizeof(x) / sizeof(double);' % color
			print 'data->%s.x = g_memdup(x, sizeof(x));' % color
			print 'data->%s.y = g_memdup(y, sizeof(y));' % color
			print '}'
		print 'return TRUE;'
		print '} else',
print 'return FALSE;'
print "}"
