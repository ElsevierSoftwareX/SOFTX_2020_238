# Copyright (C) 2015  Madeline Wade
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

__author__ = "Madeline Wade <madeline.wade@ligo.org>"

import numpy
import gst
import sys
import gobject

from gstlal import pipeio

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def constant_upsample(inbuf, outbuf, cadence):
	out = []
	in_data = numpy.frombuffer(inbuf[:], dtype=numpy.float64)
	for val in in_data:
		for i in xrange(cadence):
			out.append(val)
	out = numpy.array(out, dtype = numpy.float64)
	out_len = out.nbytes
	outbuf[:out_len] = numpy.getbuffer(out)

def compute_output_samples(inbuf, outbuf, unit_size, cadence, gap = False):
	assert (not inbuf.size % unit_size), "The input buffer size is not evenly divisble by the unit size"
	inbuf_size = inbuf.size / unit_size
	output_samples = inbuf_size * cadence
	if not gap:
		constant_upsample(inbuf, outbuf, cadence)
	return output_samples

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_constant_upsample(gst.BaseTransform):
	__gstdetails__ = (
		"Upsample stream of constant values",
		"Filter/Audio",
		"Upsamples a stream filling the upsampled samples with the same constant value as the input",
		__author__
	)

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		)
	)

	def __init__(self):
		super(lal_constant_upsample, self).__init__()
		self.rate_in = 0
		self.rate_out = 0
		self.unit_size = 0
		self.set_gap_aware(True)

	def do_get_unit_size(self, caps):
		return pipeio.get_unit_size(caps)

	def do_transform_caps(self, direction, caps):
		caps = gst.Caps.copy(caps)
		rate, = [s["rate"] for s in caps]
		if direction == gst.PAD_SRC:
			# Source pad's format is the same as sink pad's 
			# except it can have any sample rate equal to or 
			# greater than the sink pad's. (Really needs to be 
			# an integer multiple, actually, but that requirement 
			# is not enforced here).
			tmpltcaps = self.get_pad("sink").get_pad_template_caps()
			for n in range(gst.Caps.get_size(caps)):
				s = caps[n]
				if s.has_field_typed("rate", "GstIntRange"):
					if rate.high == 1:
						allowed_rate = 1
					else:
						allowed_rate = gst.IntRange(1, rate.high)
				elif s.has_field_typed("rate", "gint"):
					if rate == 1:
						allowed_rate = 1
					else:
						allowed_rate = gst.IntRange(1, rate)
				else:
					gst.message_new_error(self, gst.GError(gst.CORE_ERROR, gst.CORE_ERROR_NEGOTIATION, "negotiation error"), "invalid type for rate in caps") 
		elif direction == gst.PAD_SINK:
			# Source pad's format is the same as sink pad's 
			# except it can have any sample rate equal to or 
			# greater than the sink pad's. (Really needs to be 
			# an integer multiple, actually, but that requirement 
			# is not enforced here).
			tmpltcaps = self.get_pad("src").get_pad_template_caps()
			for n in range(gst.Caps.get_size(caps)):
				s = caps[n]
				if s.has_field_typed("rate", "GstIntRange"):
					allowed_rate = gst.IntRange(rate.low, gobject.G_MAXINT)
				elif s.has_field_typed("rate", "gint"):
					allowed_rate = gst.IntRange(rate, gobject.G_MAXINT)
				else:
					gst.message_new_error(self, gst.GError(gst.CORE_ERROR, gst.CORE_ERROR_NEGOTIATION, "negotiation error"), "invalid type for rate in caps") 
		else:
			raise AssertionError
		result = gst.Caps()
		for s in tmpltcaps:
			s = s.copy()
			s["rate"] = allowed_rate
			result.append_structure(s)
		return result	

	def do_set_caps(self, incaps, outcaps):
		# Enforce that the input rate has to be an integer multiple of the output rate
		rate_in = incaps[0]["rate"]
		rate_out = outcaps[0]["rate"]
		unit_size = self.do_get_unit_size(incaps)
		if not unit_size:
			return False
		if rate_out % rate_in:
			gst.log("output rate is not an integer multiple of input rate. input rate = %d output rate = %d" %(rate_out, rate_in))
			return False
		self.rate_in = rate_in
		self.rate_out = rate_out
		self.unit_size = unit_size
		return True
		
	def do_transform_size(self, direction, caps, size, othercaps):
		self.cadence = self.rate_out / self.rate_in
		unit_size = self.do_get_unit_size(caps)
		if not unit_size:
			return False	
		# Convert byte count to samples
		if size % unit_size:
			gst.log("buffer size %d is not a multiple of %d" % (size, unit_size))
			return False
		size /= unit_size

		if direction == gst.PAD_SRC:
			# Compute samples required on the sink pad to 
			# produce requested sample count on source pad

			#
			# size = # of samples requested on source pad
			#
			# cadence = # of output samples per input sample
			if size >= self.cadence:
				othersize = size / self.cadence
			else:
				othersize = 0
			othersize *= unit_size
			return int(othersize)
		elif direction == gst.PAD_SINK:
			# Compute samples to be produced on source pad
			# from sample count available on sink pad
			#
			# size = # of samples available on sink pad
			#
			# cadence = # of output samples per input sample
			othersize = size * self.cadence
			othersize *= unit_size
			return int(othersize)
		else:
			raise ValueError(direction)		

	def do_start(self):
		self.t0 = gst.CLOCK_TIME_NONE
		self.offset0 = gst.BUFFER_OFFSET_NONE
		self.next_in_offset = gst.BUFFER_OFFSET_NONE
		self.next_out_offset = gst.BUFFER_OFFSET_NONE
		self.need_discont = True
		self.need_gap = False
		return True

	def set_metadata(self, buf, output_samples, gap):
		buf.size = output_samples * self.unit_size
		buf.offset = self.next_out_offset
		self.next_out_offset += output_samples
		buf.offset_end = self.next_out_offset
		buf.timestamp = self.t0 + gst.util_uint64_scale_int_round(buf.offset - self.offset0, gst.SECOND, self.rate_out)
		buf.duration = self.t0 + gst.util_uint64_scale_int_round(buf.offset_end - self.offset0, gst.SECOND, self.rate_out) - buf.timestamp
		if self.need_discont:
			gst.Buffer.flag_set(buf, gst.BUFFER_FLAG_DISCONT)
			self.need_discont = False
		if gap or self.need_gap:
			gst.Buffer.flag_set(buf, gst.BUFFER_FLAG_GAP)
			if output_samples > 0:
				self.need_gap = False
		else:
			gst.Buffer.flag_unset(buf, gst.BUFFER_FLAG_GAP)
		
	def do_transform(self, inbuf, outbuf):
		# FIXME: I'm not sure this is the right fix for hearbeat buffers, so I need to check this!
		if len(inbuf) == 0:
			gst.Buffer.flag_set(inbuf, gst.BUFFER_FLAG_GAP)
		if gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_DISCONT) or inbuf.offset != self.next_in_offset or self.t0 == gst.CLOCK_TIME_NONE:
			self.t0 = inbuf.timestamp
			self.offset0 = self.next_out_offset = gst.util_uint64_scale_int_ceil(inbuf.offset, self.rate_out, self.rate_in)
			self.need_discont = True
		self.next_in_offset = inbuf.offset_end

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			output_samples = compute_output_samples(inbuf, outbuf, self.unit_size, self.rate_out / self.rate_in, False)
			self.set_metadata(outbuf, output_samples, False)
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
			output_samples = compute_output_samples(inbuf, outbuf, self.unit_size, self.rate_out / self.rate_in, True)
			self.set_metadata(outbuf, output_samples, True)
			if output_samples == 0:
				self.need_gap = True
	
		return gst.FLOW_OK

gobject.type_register(lal_constant_upsample)

__gstelementfactory__ = (
	lal_constant_upsample.__name__,
	gst.RANK_NONE,
	lal_constant_upsample
)
