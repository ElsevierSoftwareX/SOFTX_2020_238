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
	return "%d.%09d" % (timestamp // gst.SECOND, timestamp % gst.SECOND)


class CheckTimeStamps(gst.BaseTransform):
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
		gst.BaseTransform.__init__(self)
		self.set_passthrough(True)


	def do_set_caps(self, incaps, outcaps):
		self.channels = incaps[0]["channels"]
		self.width = incaps[0]["width"]
		self.rate = incaps[0]["rate"]
		return True


	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_offset = None
		self.next_timestamp = None
		return True


	def do_transform_ip(self, buf):
		#
		# initialize timestamp book-keeping
		#

		if self.t0 is None:
			self.next_timestamp = self.t0 = buf.timestamp
			self.next_offset = self.offset0 = buf.offset

		#
		# check timestamps and offsets
		#

		if buf.timestamp != self.next_timestamp:
			print >>sys.stderr, "%s: got timestamp %s expected %s" % (self.get_property("name"), printable_timestamp(buf.timestamp), printable_timestamp(self.next_timestamp))
		if buf.offset != self.next_offset:
			print >>sys.stderr, "%s: got offset %d expected %d" % (self.get_property("name"), buf.offset, self.next_offset)

		expected_offset = self.offset0 + int(round((buf.timestamp - self.t0) * float(self.rate) / gst.SECOND))
		if buf.offset != expected_offset:
			print >>sys.stderr, "%s: timestamp/offset mismatch:  at a rate of %d samples/s buffer's timestamp %s corresponds to offset %d, got %d" % (self.get_property("name"), self.rate, printable_timestamp(buf.timestamp), expected_offset, buf.offset)

		length = buf.offset_end - buf.offset
		if buf.size != length * (self.width // 8 * self.channels):
			print >>sys.stderr, "%s: buffer length %d corresponds to size %d, got %d" % (self.get_property("name"), length, length * (self.width // 8 * self.channels), buf.size)

		#
		# reset for next buffer
		#

		self.next_offset = buf.offset_end
		self.next_timestamp = buf.timestamp + buf.duration

		return gst.FLOW_OK


gobject.type_register(CheckTimeStamps)


def mkchecktimestamps(pipeline, src, name = None):
	elem = CheckTimeStamps()
	if name is not None:
		elem.set_property("name", name)
	pipeline.add(elem)
	src.link(elem)
	return elem
