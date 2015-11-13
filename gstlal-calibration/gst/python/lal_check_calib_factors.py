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

from gstlal import pipeio

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def determine_factor_value(inbuf, outbuf, var, wait_time_ns, last_best, last_best_ts, rate):
	out = []
	start_ts = inbuf.timestamp
	current = numpy.frombuffer(inbuf[:], dtype = numpy.float64)
	dt = 1/(float(rate)) * gst.SECOND
	for j, i in enumerate(current):
		current_ts = start_ts + dt
		if (i >= (last_best - var)) and (i <= (last_best + var)):
			last_best = i
			last_best_ts = current_ts
			val = i
		else:
			if (current_ts - last_best_ts > wait_time_ns) and not numpy.isnan(i) and not numpy.isinf(i):
				last_best = i
				last_best_ts = current_ts
				val = i
			else:
				val = last_best
		out.append(val)
	out = numpy.array(out, dtype = numpy.float64)
	out_len = out.nbytes
	outbuf[:out_len] = numpy.getbuffer(out)
	return last_best, last_best_ts

#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

class lal_check_calib_factors(gst.BaseTransform):
	__gstdetails__ = (
		"Check Calibration Factors",
		"Filter/Audio",
		"Checks the value of calibration factors compared to a specified minimum and maximum value.  Returns the input if it lies within expected range; returns the last best computed value if it lies outside expected range.",
		__author__
	)

	__gproperties__ = {
		"variance" : (
			gobject.TYPE_DOUBLE,
			"Variance",
			"Variance from expected for last best value of calibration factor",
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

	def do_get_property(self, prop):
		if prop.name == "variance":
			return self.var
		elif prop.name == "wait-time-to-new-expected":
			return self.wait
		elif prop.name == "default":
			return self.default

	def do_start(self):
		self.last_best = self.default
		self.last_best_ts = 0
		self.rate = 16384
		return True

	def do_set_caps(self, incaps, outcaps):
		self.rate = incaps[0]["rate"]
		return True

	def do_transform(self, inbuf, outbuf):
		# FIXME: I'm not sure this is the right fix for hearbeat buffers, so I need to check this!
		if len(inbuf) == 0:
			gst.Buffer.flag_set(inbuf, gst.BUFFER_FLAG_GAP)

		# Process buffer
		if not gst.Buffer.flag_is_set(inbuf, gst.BUFFER_FLAG_GAP):
			# Input is not 0s
			self.last_best, self.last_best_ts = determine_factor_value(inbuf, outbuf, self.var, self.wait * gst.SECOND, self.last_best, self.last_best_ts, self.rate) 
		else:
			# Input is 0s
			gst.Buffer.flag_set(outbuf, gst.BUFFER_FLAG_GAP)
	
		return gst.FLOW_OK

gobject.type_register(lal_check_calib_factors)

__gstelementfactory__ = (
	lal_check_calib_factors.__name__,
	gst.RANK_NONE,
	lal_check_calib_factors
)
