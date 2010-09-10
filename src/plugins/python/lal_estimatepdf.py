# Copyright (C) 2010 Nickolas Fotopoulos
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
Accumulate triggers, histogram them, and use the histogram to assign FAR values.
Triggers that come before the histogram is populated to min_hist_len are marked
with a FAR of inf.
"""
__author__ = "Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import collections
import sys

from gstlal.pipeutil import *
from gst.extend.pygobject import gproperty
import numpy as np

from pylal import rate
from pylal.xlal.datatypes import snglinspiraltable


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#

#
# For generality, I try to only access trigger data through these functions
# so that we can swap new ones in for different types of triggers.
#

def trigger_time(trig):
	return trig.end_time * gst.SECOND + trig.end_time_ns

hist_mchirp_bins = rate.ATanBins(1.1, 1.40, 40)
def trigger_hist_group(trig):
	return hist_mchirp_bins[trig.mchirp]

def trigger_stat(trig):
	return trig.snr

#
# Helper class
#

class MovingHistogram(object):
	def __init__(self, bins, max_hist_len):
		super(MovingHistogram, self).__init__()
		self.bins = bins
		self.max_hist_len = max_hist_len

		self.hist_ind = collections.deque()
		self.timestamps = collections.deque()
		self.hist = np.zeros(len(bins), dtype=float)

	def __len__(self):
		return len(self.hist_ind)

	def update(self, timestamp, stat):
		"""
		Push the stat's bin onto the deque and update the histogram.
		"""
		ind = self.bins[stat]
		self.hist[ind] += 1
		if len(self) >= self.max_hist_len:
			self.hist[self.hist_ind.popleft()] -= 1
			self.timestamps.popleft() # FIXME: deques can work as ring buffers in Python > 2.6
		self.hist_ind.append(ind)
		self.timestamps.append(timestamp)

	def get_count(self, stat):
		# FIXME: This may by slow with a deque. Must profile.
		return self.hist[self.bins[stat]:].sum()

	def get_livetime(self):
		# FIXME: This is a super naive livetime estimation.
		return self.timestamps[-1] - self.timestamps[0]

	def get_far(self, stat):
		"""
		Return the FAR (false-alarm rate) of the given stat based on the
		contents of the histogram.
		"""
		# Reminder: timestamps are in ns, FAR is in Hz

		# FIXME FIXME FIXME: adding 1 to count to guarantee non-zero FARs, with
		# a minum of (1/livetime), but this can really screw up FARs in the tail!
		return (self.get_count(stat) + 1) / self.get_livetime() * gst.SECOND


#
# Main element
#

class lal_estimatepdf(gst.BaseTransform):
	__gstdetails__ = (
		'Trigger statistic PDF Estimation Element',
		'Generic',
		__doc__,
		__author__
	)

	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = (int) 1
			""")
		),
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = (int) 1
			""")
		)
	)

	gproperty(
		gobject.TYPE_UINT64,
		"min-history-length",
		"The minimum number of triggers to include in the histogram",
		0, gst.CLOCK_TIME_NONE, 100,
		construct=True
	)
	gproperty(
		gobject.TYPE_UINT64,
		"max-history-length",
		"The maximum number of triggers to include in the histogram",
		0, gst.CLOCK_TIME_NONE, 100,
		construct=True
	)
	gproperty(
		gobject.TYPE_UINT64,
		"min-trigger-age",
		"The minimum age of a trigger before it can enter the histogram in nanoseconds",
		0, gst.CLOCK_TIME_NONE, 10 * gst.SECOND,
		construct=True
	)

	def __init__(self):
		super(lal_estimatepdf, self).__init__()
		self.src_pads().next().use_fixed_caps()
		self.sink_pads().next().use_fixed_caps()
		for prop in self.props:
			self.set_property(prop.name, prop.default_value)
		# FIXME: using linear bins imposes a minimum and maximum SNR.  If
		# a trigger has SNR that is greater than or less than this value, then
		# pylal.rate will actually raise an IndexError!
		self.bins = rate.ATanBins(5, 6, 1000)

		# have one moving hist per template
		self.moving_hist_dict = {}
		self.held_triggers = collections.deque()

	def do_start(self):
		self.t0 = None
		self.offset0 = None
		self.next_offset = None
		self.next_timestamp = None
		return True

	def do_transform_ip(self, buf):
		min_trigger_age = self.get_property("min-trigger-age")
		min_hist_len = self.get_property("min-history-length")
		max_hist_len = self.get_property("max-history-length")
		held = self.held_triggers
		moving_hist_dict = self.moving_hist_dict
		trigs = snglinspiraltable.from_buffer(buf)
		for i, trig in enumerate(trigs):
			# update moving histogram with held triggers that are old enough
			min_time = trigger_time(trig) - min_trigger_age
			while len(held) > 0 and trigger_time(held[0]) < min_time:
				temp_trig = held.popleft()
				temp_hist = moving_hist_dict.get(trigger_hist_group(temp_trig))
				assert temp_hist is not None  # temp_trig should already have passed through FAR assignment
				temp_hist.update(trigger_time(temp_trig), trigger_stat(temp_trig))

			# hold current trigger to be incorporated into future histograms
			held.append(trig)

			# assign FAR
			moving_hist = moving_hist_dict.get(trigger_hist_group(trig))
			if moving_hist is None:
				moving_hist = moving_hist_dict.setdefault(trigger_hist_group(trig), MovingHistogram(self.bins, max_hist_len))
			if len(moving_hist) >= min_hist_len:
				trig.alpha = moving_hist.get_far(trigger_stat(trig))
				#print >>sys.stderr, trigger_time(trig), trigger_stat(trig), trig.alpha
			else:
				trig.alpha = float("inf")  # FIXME: discard trig

			# write back to buffer
			data = buffer(trig)
			datasize = len(data)
			buf[i*datasize:(i+1)*datasize] = data
		#
		# done
		#

		return gst.FLOW_OK




# Register element class
gstlal_element_register(lal_estimatepdf)

