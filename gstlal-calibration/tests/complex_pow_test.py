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

def complex_pow_01(pipeline, name):

	buffer_length = 1.0
	rate = 2048
	width = 32
	channels = 2
	test_duration = 10.0
	freq = 0
	is_live = False


	src = pipeparts.mkaudiotestsrc(pipeline, wave = 5, freq = freq, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
	src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F%d%s, rate=%d, channels=2, width=%d, endianness=1234" % (width, BYTE_ORDER, rate, width))
	mix = numpy.random.random((2, 2)).astype("float64")
	out = pipeparts.mkmatrixmixer(pipeline, src, matrix=mix)
	out = pipeparts.mktogglecomplex(pipeline, out)
	outtee = pipeparts.mktee(pipeline, out)

	pipeparts.mknxydumpsink(pipeline, outtee, "before_pow_01.dump")

	out = pipeparts.mktogglecomplex(pipeline, outtee) 
	out = pipeparts.mkgeneric(pipeline, out, "complex_pow", exponent=2)
	out = pipeparts.mktogglecomplex(pipeline, out) 

	pipeparts.mknxydumpsink(pipeline, out, "after_pow_01.dump")

	return pipeline
	
def complex_pow_02(pipeline, name):

	buffer_length = 1.0
	rate = 2048
	width = 32
	test_duration = 10.0
	freq = 0
	is_live = False

	src = pipeparts.mkaudiotestsrc(pipeline, wave=5, freq=freq, blocksize = 8*int(buffer_length*rate), volume = 1, num_buffers = int(test_duration/buffer_length), is_live = is_live)
	src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F%d%s, rate=%d, channels=1, width=%d, channel-mask=0, endianness=1234" % (width, BYTE_ORDER, rate, width))
	tee = pipeparts.mktee(pipeline, src)
	
	out = pipeparts.mkgeneric(pipeline, None, "interleave")
	pipeparts.mkqueue(pipeline, tee).link(out)
	pipeparts.mkqueue(pipeline, tee).link(out)

	out = pipeparts.mkaudiorate(pipeline, out)
	mix = numpy.random.random((2,2)).astype("float64")
	out = pipeparts.mkmatrixmixer(pipeline, out, matrix=mix)
	out = pipeparts.mktogglecomplex(pipeline, out)	

	outtee = pipeparts.mktee(pipeline, out)
	pipeparts.mknxydumpsink(pipeline, outtee, "before_pow_02.dump")
	out = pipeparts.mkqueue(pipeline, outtee)

	out = pipeparts.mktogglecomplex(pipeline, out)
	out = pipeparts.mkgeneric(pipeline, out, "complex_pow", exponent=2)
	out = pipeparts.mktogglecomplex(pipeline, out)

	pipeparts.mknxydumpsink(pipeline, out, "after_pow_02.dump")

	return pipeline


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(complex_pow_01, "complex_pow_01")
test_common.build_and_run(complex_pow_02, "complex_pow_02")

