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
Wrapper for gobject.option, an adapter for optparse and GOption.

Implements the fixes proposed in Gnome Bugzilla at:
https://bugzilla.gnome.org/show_bug.cgi?id=564070 (courtesy of Laszlo Pandy)
"""
__author__ = "Leo Singer <leo.singer@LIGO.ORG>"


import gobject
from gobject import option as goption

OptionParser = goption.OptionParser

# This is the test that would fail if pygobject hasn't been patched.
# FIXME: Delete this when 564070 has been fixed.
a = OptionParser();
a.add_option_group(gobject.OptionGroup('foo', 'bar', 'bat', None))
try:
	a.parse_args([])
except AttributeError:
	# Subclass OptionParser, and fix the broken method.
	import optparse
	import sys
	class OptionParser(OptionParser):
		def parse_args(self, args=None, values=None):
			try:
				return optparse.OptionParser.parse_args(self, args, values)
			except gobject.GError:
				error = sys.exc_info()[1]
				if error.domain != gobject.OPTION_ERROR:
					raise
				if error.code == gobject.OPTION_ERROR_BAD_VALUE:
					raise goption.OptionValueError(error.message)
				elif error.code == gobject.OPTION_ERROR_UNKNOWN_OPTION:
					raise goption.BadOptionError(error.message)
				elif error.code == gobject.OPTION_ERROR_FAILED:
					raise goption.OptParseError(error.message)
				else:
					raise
del a
