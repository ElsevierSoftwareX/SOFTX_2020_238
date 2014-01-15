# Copyright (C) 2014  Madeline Wade, Chris Pankow
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

def logical_op(inbuf, outbuf, cadence, output_samples, new_remainder, required_on, status_out, old_leftover, old_remainder, data_type):
	out = []
	leftover = 0 # This is maybe confusing as a default value, since leftover could actually equal zero.  However, leftover has no meaning unless remainder is also nonzero, and leftover will be properly set below if remainder is nonzero.
	old_bits = numpy.zeros(old_remainder, dtype = data_type)
	new_bits = numpy.frombuffer(inbuf[:], dtype = data_type)
	old_bits.fill(old_leftover)
	inbits = numpy.concatenate((old_bits, new_bits))
	if output_samples > 0:
		for i in xrange(output_samples):
			if (reduce(lambda x, y: x & y, inbits[i*cadence:i*cadence+cadence])) & required_on == required_on:
				out.append(status_out)
			else:
				out.append(0x0)
		out = numpy.array(out, dtype = data_type)
		out_len = out.nbytes
		outbuf[:out_len] = numpy.getbuffer(out)
	if new_remainder != 0:
		leftover = reduce(lambda x, y: x & y, inbits[-new_remainder:]) & required_on
	return leftover

def logical_undersample(inbuf, outbuf, unit_size, cadence, required_on, status_out, data_type, old_leftover = 0, old_remainder = 0, gap = False):
	assert (not inbuf.size % unit_size), "The input buffer size is not evenly divisble by the unit size"
	# Determine number of blocks and the samples leftover
	inbuf_size = inbuf.size / unit_size
	output_samples = (inbuf_size + old_remainder) / cadence
	new_remainder = (inbuf_size + old_remainder) % cadence

	if not gap:
		leftover = logical_op(inbuf, outbuf, cadence, output_samples, new_remainder, required_on, status_out, old_leftover, old_remainder, data_type)
	if gap:
		leftover = 0 

	return output_samples, leftover, new_remainder

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_logical_undersampler(gst.BaseTransform):
	__gstdetails__ = (
		"Logical Undersampler",
		"Filter/Audio",
		"Undersamples an integer stream. The undersampling applies a bit mask across all cadence samples.  (Cadence samples are the input samples that are combined to make one output sample.) The undersampled stream is therefore a summary of the cadence samples.  This element's output sample rate must be an integer of its input sample rate.",
		__author__
	)

	__gproperties__ = {
		"required-on" : (
			gobject.TYPE_UINT,
			"On bits",
			"Bit mask setting the bits that must be on in the incoming stream.",
			0, gobject.G_MAXUINT, 0x1,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"status-out" : (
			gobject.TYPE_UINT,
			"Output bits",
			"Value of output if required-on mask is true.",
			0, gobject.G_MAXUINT, 0x1,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32, " +
				"signed = (bool) {true, false}"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32, " +
				"signed = false"
			)
		)
	)

	def __init__(self):
		super(lal_logical_undersampler, self).__init__()
		self.rate_in = 0
		self.rate_out = 0
		self.unit_size = 0
		self.set_gap_aware(True)

	def do_set_property(self, prop, val):
		if prop.name == "required-on":
			self.required_on = val
		elif prop.name =="status-out":
			self.status_out = val

	def do_get_property(self, prop):
		if prop.name == "required-on":
			return self.required_on
		elif prop.name == "status-out":
			return self.status_out

	def do_get_unit_size(self, caps):
		return pipeio.get_unit_size(caps)

	def do_transform_caps(self, direction, caps):
		caps = gst.Caps.copy(caps)
		rate, = [s["rate"] for s in caps]
		if direction == gst.PAD_SRC:
			# Source pad's format is the same as sink pad's 
			# except it can have any sample rate equal to or 
			# less than the sink pad's. (Really needs to be 
			# an integer multiple, actually, but that requirement 
			# is not enforced here).
			tmpltcaps = self.get_pad("sink").get_pad_template_caps()
			for n in range(gst.Caps.get_size(caps)):
				s = caps[n]
				if s.has_field_typed("rate", "GstIntRange"):
					allowed_rate = gst.IntRange(rate.low, gobject.G_MAXINT)
				elif s.has_field_typed("rate", "gint"):
					allowed_rate = gst.IntRange(rate, gobject.G_MAXINT)
				else:
					gst.message_new_error(self, gst.GError(gst.CORE_ERROR, gst.CORE_ERROR_NEGOTIATION, "negotiation error"), "invalid type for rate in caps") 
		elif direction == gst.PAD_SINK:
			# Source pad's format is the same as sink pad's 
			# except it can have any sample rate equal to or 
			# less than the sink pad's. (Really needs to be 
			# an integer multiple, actually, but that requirement 
			# is not enforced here).
			tmpltcaps = self.get_pad("src").get_pad_template_caps()
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
		if rate_in % rate_out:
			gst.log("input rate is not an integer multiple of output rate. input rate = %d output rate = %d" %(rate_out, rate_in))
			return False
		self.rate_in = rate_in
		self.rate_out = rate_out
		self.unit_size = unit_size
		if not incaps[0]["signed"]:
			self.data_type = numpy.uint32
		elif incaps[0]["signed"]:
			self.data_type = numpy.uint32
		else: 
			return False
		return True
		
	def do_transform_size(self, direction, caps, size, othercaps):
		self.cadence = self.rate_in / self.rate_out
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
			# cadence = # of input samples per output sample
			#
			# remainder = how many extra samples of input are
			# present on the sink pad
			othersize = size * self.cadence + self.remainder # FIXME: I think this is right, but I'm not sure...
			othersize *= unit_size
			return int(othersize)
		elif direction == gst.PAD_SINK:
			# Compute samples to be produced on source pad
			# from sample count available on sink pad
			#
			# size = # of samples available on sink pad
			#
			# cadence = # of input samples per output sample
			#
			# remainder = how many extra input samples have been
			# stored because the most recent input buffer
			# ended before a complete cycle
			if size >= self.cadence - self.remainder:
				othersize = (size + self.remainder) / self.cadence
			else:
				othersize = 0
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
		self.remainder = 0
		self.leftover = 0
		self.data_type = None
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
			self.remainder = 0
			self.leftover = 0
		self.next_in_offset = inbuf.offset_end

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			output_samples, self.leftover, self.remainder = logical_undersample(inbuf, outbuf, self.unit_size, self.rate_in / self.rate_out, self.required_on, self.status_out, self.data_type, self.leftover, self.remainder, False)
			self.set_metadata(outbuf, output_samples, False)
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
			output_samples, self.leftover, self.remainder = logical_undersample(inbuf, outbuf, self.unit_size, self.rate_in / self.rate_out, self.required_on, self.status_out, self.data_type, self.leftover, self.remainder, True)
			self.set_metadata(outbuf, output_samples, True)
			if output_samples == 0:
				self.need_gap = True
	
		return gst.FLOW_OK

gobject.type_register(lal_logical_undersampler)

__gstelementfactory__ = (
	lal_logical_undersampler.__name__,
	gst.RANK_NONE,
	lal_logical_undersampler
)
