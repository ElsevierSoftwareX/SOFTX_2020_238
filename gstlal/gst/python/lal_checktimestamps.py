# Copyright (C) 2009  Kipp Cannon
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
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import sys


import pygtk
pygtk.require("2.0")
import gobject
import pygst
pygst.require('0.10')
import gst


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


def printable_timestamp(timestamp):
	return "%d.%09d s" % (timestamp // gst.SECOND, timestamp % gst.SECOND)


class lal_checktimestamps(gst.BaseTransform):
	__gstdetails__ = (
		"Timestamp Checker Pass-Through Element",
		"Generic",
		"Checks that timestamps and offsets of audio streams advance as expected and remain synchronized to each other",
		__author__
	)

	__gproperties__ = {
		"timestamp-fuzz": (
			gobject.TYPE_UINT64,
			"timestamp fuzz",
			"Number of nanoseconds of timestamp<-->offset discrepancy to accept before reporting it.  Timestamp<-->offset discrepancies of 1/2 a sample or more are always reported.",
			# FIXME:  why isn't G_MAXUINT64 defined in 2.18?
			#0, gobject.G_MAXUINT64, 1,
			0, 18446744073709551615L, 1,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"silent": (
			gobject.TYPE_BOOLEAN,
			"silent",
			"Only report errors.",
			False,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {32, 64};" +
				"audio/x-raw-complex, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {64, 128};" +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 16," +
				"depth = (int) 16," +
				"signed = (bool) {true, false}; " +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32," +
				"depth = (int) 32," +
				"signed = (bool) {true, false}; " +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64," +
				"depth = (int) 64," +
				"signed = (bool) {true, false}"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {32, 64};" +
				"audio/x-raw-complex, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) {64, 128};" +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 16," +
				"depth = (int) 16," +
				"signed = (bool) {true, false}; " +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32," +
				"depth = (int) 32," +
				"signed = (bool) {true, false}; " +
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64," +
				"depth = (int) 64," +
				"signed = (bool) {true, false}"
			)
		)
	)


	def __init__(self):
		super(lal_checktimestamps, self).__init__()
		self.set_passthrough(True)


	def do_set_property(self, prop, val):
		if prop.name == "timestamp-fuzz":
			self.timestamp_fuzz = val
		elif prop.name == "silent":
			self.silent = val


	def do_get_property(self, prop):
		if prop.name == "timestamp-fuzz":
			return self.timestamp_fuzz
		elif prop.name == "silent":
			return self.silent


	def do_set_caps(self, incaps, outcaps):
		self.unit_size = incaps[0]["width"] // 8 * incaps[0]["channels"]
		self.units_per_second = incaps[0]["rate"]
		return True


	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_offset = None
		self.next_timestamp = None
		return True


	def check_time_offset_mismatch(self, buf):
		expected_offset = self.offset0 + int(round((buf.timestamp - self.t0) * float(self.units_per_second) / gst.SECOND))
		expected_timestamp = self.t0 + int(round((buf.offset - self.offset0) * gst.SECOND / float(self.units_per_second)))
		if buf.offset != expected_offset:
			print >>sys.stderr, "%s: timestamp/offset mismatch%s:  got offset %d, buffer timestamp %s corresponds to offset %d (error = %d samples)" % (self.get_property("name"), (buf.flag_is_set(gst.BUFFER_FLAG_DISCONT) and " at discontinuity" or ""), buf.offset, printable_timestamp(buf.timestamp), expected_offset, buf.offset - expected_offset)
		elif abs(buf.timestamp - expected_timestamp) > self.timestamp_fuzz:
			print >>sys.stderr, "%s: timestamp/offset mismatch%s:  got timestamp %s, buffer offset %d corresponds to timestamp %s (error = %d ns)" % (self.get_property("name"), (buf.flag_is_set(gst.BUFFER_FLAG_DISCONT) and " at discontinuity" or ""), printable_timestamp(buf.timestamp), buf.offset, printable_timestamp(expected_timestamp), buf.timestamp - expected_timestamp)


	def do_transform_ip(self, buf):
		if self.t0 is None or buf.flag_is_set(gst.BUFFER_FLAG_DISCONT):
			if self.t0 is None:
				if not self.silent:
					print >>sys.stderr, "%s: initial timestamp = %s, offset = %d" % (self.get_property("name"), printable_timestamp(buf.timestamp), buf.offset)
			elif buf.flag_is_set(gst.BUFFER_FLAG_DISCONT):
				print >>sys.stderr, "%s: discontinuity:  timestamp = %s, offset = %d;  would have been %s, offset = %d" % (self.get_property("name"), printable_timestamp(buf.timestamp), buf.offset, printable_timestamp(self.next_timestamp), self.next_offset)

				#
				# check for timestamp/offset mismatch
				#

				self.check_time_offset_mismatch(buf)

			#
			# reset/initialize timestamp book-keeping
			#

			self.next_timestamp = self.t0 = buf.timestamp
			self.next_offset = self.offset0 = buf.offset
		else:
			#
			# check for timestamp/offset discontinuities
			#

			if buf.timestamp != self.next_timestamp:
				print >>sys.stderr, "%s: got timestamp %s expected %s (discont flag is %s)" % (self.get_property("name"), printable_timestamp(buf.timestamp), printable_timestamp(self.next_timestamp), buf.flag_is_set(gst.BUFFER_FLAG_DISCONT) and "set" or "not set")
			if buf.offset != self.next_offset:
				print >>sys.stderr, "%s: got offset %d expected %d (discont flag is %s)" % (self.get_property("name"), buf.offset, self.next_offset, buf.flag_is_set(gst.BUFFER_FLAG_DISCONT) and "set" or "not set")

			#
			# check for timestamp/offset mismatch
			#

			self.check_time_offset_mismatch(buf)

		#
		# check for buffer size / sample count mismatch
		#

		length = buf.offset_end - buf.offset
		allowed_sizes = [length * self.unit_size]
		if buf.flag_is_set(gst.BUFFER_FLAG_GAP):
			allowed_sizes.append(0)
		if buf.size not in allowed_sizes:
			print >>sys.stderr, "%s: got buffer size %d, buffer length %d corresponds to size %d" % (self.get_property("name"), buf.size, length, length * self.unit_size)

		#
		# reset for next buffer
		#

		self.next_offset = buf.offset_end
		self.next_timestamp = buf.timestamp + buf.duration

		#
		# done
		#

		return gst.FLOW_OK


#
# register element class
#


gobject.type_register(lal_checktimestamps)

__gstelementfactory__ = (
	lal_checktimestamps.__name__,
	gst.RANK_NONE,
	lal_checktimestamps
)
