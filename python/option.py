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
https://bugzilla.gnome.org/show_bug.cgi?id=627449 (contributed by me)
"""
__author__ = "Leo Singer <leo.singer@LIGO.ORG>"


# FIXME: Delete this when 627449 has been fixed.
try:
	from gobject.option import *
except:
	from gobject.option import OptParseError, OptionError, OptionValueError, BadOptionError, OptionConflictError, Option, OptionGroup, OptionParser, make_option


# FIXME: Delete this when 564070 has been fixed.
import gobject as _gobject
try:
	# This is the test that would fail if pygobject hasn't been patched.
	a = OptionParser();
	a.add_option_group(_gobject.OptionGroup('foo', 'bar', 'bat', None))
	a.parse_args([])
	del a
except:
	import sys
	import optparse
	# Subclass OptionParser, and fix the broken method.
	class NewOptionParser(OptionParser):
		def parse_args(self, args=None, values=None):
			try:
				options, args = optparse.OptionParser.parse_args(
					self, args, values)
				return options, args
			except _gobject.GError:
				error = sys.exc_info()[1]
				if error.domain != _gobject.OPTION_ERROR:
					raise
				if error.code == _gobject.OPTION_ERROR_BAD_VALUE:
					raise OptionValueError(error.message)
				elif error.code == _gobject.OPTION_ERROR_UNKNOWN_OPTION:
					raise BadOptionError(error.message)
				elif error.code == _gobject.OPTION_ERROR_FAILED:
					raise OptParseError(error.message)
				else:
					raise
	OptionParser = NewOptionParser
