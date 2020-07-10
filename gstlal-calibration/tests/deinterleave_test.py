#!/usr/bin/env python3
# Copyright (C) 2013  Madeline Wade
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

import numpy
import sys
from gstlal import pipeparts
from gstlal import calibration_parts
import test_common

if sys.byteorder == "little":
	BYTE_ORDER = "LE"
else:
	BYTE_ORDER = "BE"


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#

def deinterleave_01(pipeline, name):

	buffer_length = 1.0
	rate = 2048
	width = 64
	channels = 2
	test_duration = 10.0
	freq = 0
	is_live = False


	src = pipeparts.mkaudiotestsrc(pipeline, wave = 5, freq = freq, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
	src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F%d%s, rate=%d, channels=2, layout=interleaved" % (width, BYTE_ORDER, rate))
	
	elem = pipeparts.mkgeneric(pipeline, src, "deinterleave")
	out1 = pipeparts.mkqueue(pipeline, None)
	out2 = pipeparts.mkqueue(pipeline, None)
	pipeparts.src_deferred_link(elem, "src_0", out1.get_static_pad("sink"))
	pipeparts.src_deferred_link(elem, "src_1", out2.get_static_pad("sink"))
	#out1 = pipeparts.mkaudioconvert(pipeline, out1)
	out1 = pipeparts.mkcapsfilter(pipeline, out1, "audio/x-raw, format=F%d%s, rate=%d, channels=1, layout=interleaved" % (width, BYTE_ORDER, rate))
	#out2 = pipeparts.mkaudioconvert(pipeline, out2)
	out2 = pipeparts.mkcapsfilter(pipeline, out2, "audio/x-raw, format=F%d%s, rate=%d, channels=1, layout=interleaved" % (width, BYTE_ORDER, rate))
	
	pipeparts.mknxydumpsink(pipeline, out1, "out1.dump")
	pipeparts.mknxydumpsink(pipeline, out2, "out2.dump")
	#pipeparts.mkfakesink(pipeline, out1)
	#pipeparts.mkfakesink(pipeline, out2)
	
	return pipeline

def deinterleave_02(pipeline, name):

	buffer_length = 1.0
        rate = 2048
        width = 64
        channels = 3
        test_duration = 10.0
        freq = 10
        is_live = False


	src = test_common.test_src(pipeline, wave = 5, freq = freq, test_duration = test_duration, volume = 1, width = 64, rate = rate, channels = channels, verbose = False)
	streams = calibration_parts.mkdeinterleave(pipeline, src, channels)
	for i in range(0, channels):
		pipeparts.mknxydumpsink(pipeline, streams[i], "%s_stream%d.txt" % (name, i))

	return pipeline

#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


#test_common.build_and_run(deinterleave_01, "deinterleave_01")
test_common.build_and_run(deinterleave_02, "deinterleave_02")

