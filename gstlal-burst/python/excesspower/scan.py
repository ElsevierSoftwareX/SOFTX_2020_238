# Copyright (C) 2014 Chris Pankow
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
"""Module holding utilities and objects for an excesspower scan"""

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import os
import tempfile

import numpy

from glue.segments import segment
from glue.lal import LIGOTimeGPS, Cache, CacheEntry

from gstlal import pipeparts
from gstlal.excesspower.utils import EXCESSPOWER_UNIT_SCALE
from gstlal.excesspower.parts import mknxyfdsink

#
# =============================================================================
#
#                                Scan Class
#
# =============================================================================L
#

class EPScan(object):
	def __init__(self, scan_segment, low_freq, high_freq, base_band):
		self.serializer_dict = {}
		self.scan_segment = scan_segment
		self.bandwidth = segment(low_freq, high_freq)
		self.base_band = base_band

	def add_data_sink(self, pipeline, head, name, type, units=EXCESSPOWER_UNIT_SCALE['Hz']):
		mknxyfdsink(pipeline,
			pipeparts.mkqueue(pipeline, head),
			self.get_tmp_fd(name, type),
			self.scan_segment,
			units
		)

	def get_tmp_fd(self, name, type):
		"""
        Create a temporary file and file descriptor, returning the descriptor... mostly for use with fdsink. Name is an internal identifier and 'write_out' will move the temporary file to this name.
		"""
		tmpfile, tmpname = tempfile.mkstemp()
		self.serializer_dict[name] = (tmpfile, tmpname)
		return tmpfile

	def write_out(self, scan_name):
		"""
		Move all temporary files to their permanent homes. Note that this clears the internal dictionary of filenames / contents.
		"""
		for name, (fd, fname) in self.serializer_dict.iteritems():
			self.serializer_dict[name] = numpy.loadtxt(fname)
		self.serializer_dict["segment"] = self.scan_segment
		self.serializer_dict["bandwidth"] = list(self.bandwidth) + [self.base_band]
		# FIXME: Reenable when it becomes available
		#numpy.savez_compressed(scan_name, **self.serializer_dict)
		numpy.savez(scan_name, **self.serializer_dict)
		self.serializer_dict.clear()

	def close(self):
		"""
		Close all temporary files.
		"""
		for (fd, fname) in self.serializer_dict.values():
			os.close(fd)
