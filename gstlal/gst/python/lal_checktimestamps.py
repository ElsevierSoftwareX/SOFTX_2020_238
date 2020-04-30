# Copyright (C) 2009--2013  Kipp Cannon
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


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstAudio', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import GObject
from gi.repository import Gst
from gi.repository import GstAudio
from gi.repository import GstBase


GObject.threads_init()
Gst.init(None)


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


## @file lal_checktimestamps.py
# This gstreamer element checks timestamps; see lal_checktimestamps for more information


## @package lal_checktimestamps
# A gstreamer element to check time stamps
#
# ### Review status
# - Code walkthrough 2014/02/12 
# - Review git hash: ecaf8840ada2877b9b2a8a144a62874c004cd3d2
# - Folks involved J. Creighton, B.S. Sathyaprakash, K. Cannon, C. Hanna, F. Robinet
#
#


def printable_timestamp(timestamp):
	"""!
	A function to nicely format a timestamp for printing
	"""
	if timestamp is None or timestamp == Gst.CLOCK_TIME_NONE:
		return "(none)"
	return "%d.%09d s" % (timestamp // Gst.SECOND, timestamp % Gst.SECOND)


def gst_buffer_flag_is_set(buf, flags):
	# FIXME:  this should not be needed.  figure out how
	# GST_BUFFER_FLAG_IS_SET() is exported via gir
	return buf.mini_object.flags & flags == flags


class lal_checktimestamps(GstBase.BaseTransform):
	"""!
	Gstreamer element that will verify that the timestamps agree with
	incoming buffers based on tracking the buffer offsets.
	"""
	__gstmetadata__ = (
		"Timestamp Checker Pass-Through Element",
		"Generic",
		"Checks that timestamps and offsets of audio streams advance as expected and remain synchronized to each other",
		__author__
	)

	#silent = GObject.Property(type = GObject.TYPE_BOOLEAN, default = False, nick = "silent", blurb = "Only report errors.", flags = GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT)
	#timestamp_fuzz = GObject.Property(type = GObject.TYPE_INT64, minimum = 0, maximum = GObject.G_MAXINT64, default = 1, nick = "timestamp fuzz", blurb = "Number of nanoseconds of timestamp<-->offset discrepancy to accept before reporting it.  Timestamp<-->offset discrepancies of 1/2 a sample or more are always reported.", flags = GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT)

	__gproperties__ = {
		"silent": (
			GObject.TYPE_BOOLEAN,
			"silent",
			"Only report errors.",
			True,
			GObject.ParamFlags.READWRITE
		),
		"timestamp-fuzz": (
			GObject.TYPE_INT64,
			"timestamp fuzz",
			"Number of nanoseconds of timestamp<-->offset discrepancy to accept before reporting it.  Timestamp<-->offset discrepancies of 1/2 a sample or more are always reported.",
			0, GObject.G_MAXINT64, 1,
			GObject.ParamFlags.READWRITE
		)
	}

	__gsttemplates__ = (
		Gst.PadTemplate.new("sink",
			Gst.PadDirection.SINK,
			Gst.PadPresence.ALWAYS,
			Gst.caps_from_string(
				"audio/x-raw, " +
				"rate = " + GstAudio.AUDIO_RATE_RANGE + ", " +
				"channels = " + GstAudio.AUDIO_CHANNELS_RANGE + ", " +
				"format = (string) { Z64LE, Z64BE, Z128LE, Z128BE }, " +
				"layout = (string) interleaved, " +
				"channel-mask = (bitmask) 0; " +
				"audio/x-raw, " +
				"rate = " + GstAudio.AUDIO_RATE_RANGE + ", " +
				"channels = " + GstAudio.AUDIO_CHANNELS_RANGE + ", " +
				"format = " + GstAudio.AUDIO_FORMATS_ALL + ", " +
				"layout = (string) interleaved, " +
				"channel-mask = (bitmask) 0"
			)
		),
		Gst.PadTemplate.new("src",
			Gst.PadDirection.SRC,
			Gst.PadPresence.ALWAYS,
			Gst.caps_from_string(
				"audio/x-raw, " +
				"rate = " + GstAudio.AUDIO_RATE_RANGE + ", " +
				"channels = " + GstAudio.AUDIO_CHANNELS_RANGE + ", " +
				"format = (string) { Z64LE, Z64BE, Z128LE, Z128BE }, " +
				"layout = (string) interleaved, " +
				"channel-mask = (bitmask) 0; " +
				"audio/x-raw, " +
				"rate = " + GstAudio.AUDIO_RATE_RANGE + ", " +
				"channels = " + GstAudio.AUDIO_CHANNELS_RANGE + ", " +
				"format = " + GstAudio.AUDIO_FORMATS_ALL + ", " +
				"layout = (string) interleaved, " +
				"channel-mask = (bitmask) 0"
			)
		)
	)


	def __init__(self):
		super(lal_checktimestamps, self).__init__()
		self.set_passthrough(True)
		self.silent = True
		self.timestamp_fuzz = 1


	def do_set_property(self, prop, val):
		if prop.name == "silent":
			self.silent = val
		elif prop.name == "timestamp-fuzz":
			self.timestamp_fuzz = val
		else:
			raise AttributeError(prop.name)


	def do_get_property(self, prop):
		if prop.name == "silent":
			return self.silent
		if prop.name == "timestamp-fuzz":
			return self.timestamp_fuzz
		raise AttributeError(prop.name)


	def do_set_caps(self, incaps, outcaps):
		info = GstAudio.AudioInfo()
		if info.from_caps(incaps):
			self.unit_size = info.bpf
			self.units_per_second = info.rate
			return True
		s = incaps.get_structure(0)
		if not s or s.get_name() != "audio/x-raw":
			return False
		success, chnls = s.get_int("channels")
		if success:
			success, rate = s.get_int("rate")
		if not success:
			return False
		fmt = s.get_string("format")
		if fmt == "Z64LE":
			self.unit_size = 8 * chnls
			self.units_per_second = rate
			return True
		elif fmt == "Z128LE":
			self.unit_size = 16 * chnls
			self.units_per_second = rate
			return True
		return False


	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_offset = None
		self.next_timestamp = None
		return True


	def check_time_offset_mismatch(self, buf):
		expected_offset = self.offset0 + int(round((buf.pts - self.t0) * float(self.units_per_second) / Gst.SECOND))
		expected_timestamp = self.t0 + int(round((buf.offset - self.offset0) * Gst.SECOND / float(self.units_per_second)))
		if buf.offset != expected_offset:
			print("%s: timestamp/offset mismatch%s:  got offset %d, buffer timestamp %s corresponds to offset %d (error = %d samples)" % (self.get_property("name"), (" at discontinuity" if gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT) else ""), buf.offset, printable_timestamp(buf.pts), expected_offset, buf.offset - expected_offset), file=sys.stderr)
		elif abs(buf.pts - expected_timestamp) > self.timestamp_fuzz:
			print("%s: timestamp/offset mismatch%s:  got timestamp %s, buffer offset %d corresponds to timestamp %s (error = %d ns)" % (self.get_property("name"), (" at discontinuity" if gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT) else ""), printable_timestamp(buf.pts), buf.offset, printable_timestamp(expected_timestamp), buf.pts - expected_timestamp), file=sys.stderr)

	def do_transform_ip(self, buf):
		if self.t0 is None or gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT):
			if self.t0 is None:
				if not self.silent:
					print >>sys.stderr, "%s: initial timestamp = %s, offset = %d" % (self.get_property("name"), printable_timestamp(buf.pts), buf.offset)
			elif gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT):
				print >>sys.stderr, "%s: DISCONT buffer:  timestamp = %s, offset = %d;  would have been %s, offset = %d" % (self.get_property("name"), printable_timestamp(buf.pts), buf.offset, printable_timestamp(self.next_timestamp), self.next_offset)

				#
				# check for timestamp/offset mismatch
				#

				self.check_time_offset_mismatch(buf)

			#
			# reset/initialize timestamp book-keeping
			#

			self.next_timestamp = self.t0 = buf.pts
			self.next_offset = self.offset0 = buf.offset
		else:
			#
			# check for timestamp/offset discontinuities
			#

			if buf.pts != self.next_timestamp:
				print >>sys.stderr, "%s: got timestamp %s expected %s (discont flag is %s)" % (self.get_property("name"), printable_timestamp(buf.pts), printable_timestamp(self.next_timestamp), "set" if gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT) else "not set")
			if buf.offset != self.next_offset:
				print >>sys.stderr, "%s: got offset %d expected %d (discont flag is %s)" % (self.get_property("name"), buf.offset, self.next_offset, "set" if gst_buffer_flag_is_set(buf, Gst.BufferFlags.DISCONT) else "not set")

			#
			# check for timestamp/offset mismatch
			#

			self.check_time_offset_mismatch(buf)

		#
		# check for buffer size / sample count mismatch
		#

		length = buf.offset_end - buf.offset
		allowed_sizes = [length * self.unit_size]
		if gst_buffer_flag_is_set(buf, Gst.BufferFlags.GAP):
			allowed_sizes.append(0)
		if buf.get_size() not in allowed_sizes:
			print >>sys.stderr, "%s: got buffer size %d, buffer length %d corresponds to size %d" % (self.get_property("name"), buf.get_size(), length, length * self.unit_size)

		#
		# reset for next buffer
		#

		self.next_offset = buf.offset_end
		self.next_timestamp = buf.pts + buf.duration

		#
		# done
		#

		return Gst.FlowReturn.OK


#
# register element class
#


GObject.type_register(lal_checktimestamps)

__gstelementfactory__ = (
	lal_checktimestamps.__name__,
	Gst.Rank.NONE,
	lal_checktimestamps
)
