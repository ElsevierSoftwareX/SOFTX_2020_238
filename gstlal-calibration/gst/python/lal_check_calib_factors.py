# Copyright (C) 2015  Madeline Wade, Chris Pankow
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
from bisect import insort_left
from numpy import floor

from gstlal import pipeio

#
# =============================================================================
#
#                                  Classes
#
# =============================================================================
#

class Running_median_array:
	def __init__(self, size):
		self.max_size = size
		self.current_size = 0
		self.array = []
		self.fifo_array = []

	def add_and_return_median(self, new_data):
		# If we're not at max, just add data; don't need to remove
		if self.current_size < self.max_size:
			insort_left(self.array, new_data)
			self.fifo_array.append(new_data)
			self.current_size = self.current_size + 1
			return self.array[int(floor(self.current_size/2))]
		else:
			self.array.remove(self.fifo_array.pop(1))
			insort_left(self.array, new_data)
			self.fifo_array.append(new_data)
			return self.array[int(floor(self.max_size/2))]	

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def determine_factor_value(inbuf, outbuf, var, wait_time_ns, last_best, last_best_ts, rate, running_median_array, statevector):
	out = []
	start_ts = inbuf.timestamp
	current = numpy.frombuffer(inbuf[:], dtype = numpy.float64)
	dt = 1/float(rate) * gst.SECOND
	for j, i in enumerate(current):
		current_ts = start_ts + j * dt
		diff = abs(i - last_best)
		if diff <= var:
			last_best = i
			last_best_ts = current_ts
			if statevector:
				val = 1.0
			else:
				val = i
		else:
			if (current_ts - last_best_ts > wait_time_ns) and not numpy.isnan(i) and not numpy.isinf(i):
				last_best = i
				last_best_ts = current_ts
				if statevector:
					val = 1.0
				else:
					val = i
			else:
				if statevector:
					val = 0.0
				else:
					val = last_best
		if statevector:
			out.append(val)
		else:
			out.append(running_median_array.add_and_return_median(val))
	
	out = numpy.array(out, dtype = numpy.float64)
	output_samples = len(out)
	out_len = out.nbytes
	outbuf[:out_len] = numpy.getbuffer(out)
	return last_best, last_best_ts, output_samples, running_median_array

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_check_calib_factors(gst.BaseTransform):
	__gstdetails__ = (
		"Smooth factors computation",
		"Filter/Audio",
		"Checks the value of calibration factors have not changed by more than a specified amount from the last good computed value. Then computes the median value over the last N samples.",
		__author__
	)

	__gproperties__ = {
		"variance" : (
			gobject.TYPE_DOUBLE,
			"Variance",
			"Variance allowed from last best value of calibration factor",
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, 0.05,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"wait-time-to-new-expected" : (
			gobject.TYPE_DOUBLE,
			"Wait time",
			"Time (in seconds) to wait until a new expected value is assigned based on current trend",
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, 1,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"default" : (
			gobject.TYPE_DOUBLE,
			"Default",
			"Default value for channel to take until acceptable value is computed.",
			-gobject.G_MAXDOUBLE, gobject.G_MAXDOUBLE, 1,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"median-array-size" : (
			gobject.TYPE_UINT,
			"Median array size",
			"Size of median array for smoothing",
			0, gobject.G_MAXUINT, 2048,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"state-vector-mode" : (
			gobject.TYPE_BOOLEAN,
			"State vector mode",
			"Produce state vector of 1s and 0s instead of smoothed kappas",
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
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64 "
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
				"width = (int) 64 "
			)
		)
	)

	def __init__(self):
		super(lal_check_calib_factors, self).__init__()
		self.set_gap_aware(True)

	def do_set_property(self, prop, val):
		if prop.name == "variance":
			self.var = val
		elif prop.name == "wait-time-to-new-expected":
			self.wait = val
		elif prop.name == "default":
			self.default = val
		elif prop.name == "median-array-size":
			self.running_median_array = Running_median_array(int(val))
		elif prop.name == "state-vector-mode":
			self.statevector = val

	def do_get_property(self, prop):
		if prop.name == "variance":
			return self.var
		elif prop.name == "wait-time-to-new-expected":
			return self.wait
		elif prop.name == "default":
			return self.default
		elif prop.name == "median-array-size":
			return self.running_median_array.max_size
		elif prop.name == "state-vector-mode":
			return self.statevector

	def do_get_unit_size(self, caps):
		return pipeio.get_unit_size(caps)

	def do_start(self):
		self.last_best = self.default
		self.last_best_ts = 0
		self.rate = None
		self.t0 = gst.CLOCK_TIME_NONE
		self.offset0 = gst.BUFFER_OFFSET_NONE
		self.next_in_offset = gst.BUFFER_OFFSET_NONE
		self.need_discont = True
		self.median_val = None
		return True

	def set_metadata(self, buf, output_samples, gap):
		buf.size = output_samples * self.unit_size
		buf.offset = self.next_out_offset
		self.next_out_offset += output_samples
		buf.offset_end = self.next_out_offset
		buf.timestamp = self.t0 + gst.util_uint64_scale_int_round(buf.offset - self.offset0, gst.SECOND, self.rate)
		buf.duration = self.t0 + gst.util_uint64_scale_int_round(buf.offset_end - self.offset0, gst.SECOND, self.rate) - buf.timestamp
		if self.need_discont:
			gst.Buffer.flag_set(buf, gst.BUFFER_FLAG_DISCONT)
			self.need_discont = False
		if gap:
			gst.Buffer.flag_set(buf, gst.BUFFER_FLAG_GAP)
		else:
			gst.Buffer.flag_unset(buf, gst.BUFFER_FLAG_GAP)


	def do_set_caps(self, incaps, outcaps):
		self.rate = incaps[0]["rate"]
		unit_size = self.do_get_unit_size(incaps)
		if not unit_size:
			return False
		self.unit_size = unit_size	
		return True

	def do_transform(self, inbuf, outbuf):
		# FIXME: I'm not sure this is the right fix for hearbeat buffers, so I need to check this!
		if len(inbuf) == 0:
			gst.Buffer.flag_set(inbuf, gst.BUFFER_FLAG_GAP)
		if gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_DISCONT) or inbuf.offset != self.next_in_offset or self.t0 == gst.CLOCK_TIME_NONE:
			self.t0 = inbuf.timestamp
			self.offset0 = self.next_out_offset = inbuf.offset
			self.need_discont = True
		self.next_in_offset = inbuf.offset_end

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			self.last_best, self.last_best_ts, output_samples, self.running_median_array = determine_factor_value(inbuf, outbuf, self.var, self.wait * gst.SECOND, self.last_best, self.last_best_ts, self.rate, self.running_median_array, self.statevector) 
			self.set_metadata(outbuf, output_samples, False)
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
			output_samples = 0
			self.set_metadata(outbuf, output_samples, True)
	
		return gst.FLOW_OK

gobject.type_register(lal_check_calib_factors)

__gstelementfactory__ = (
	lal_check_calib_factors.__name__,
	gst.RANK_NONE,
	lal_check_calib_factors
)
