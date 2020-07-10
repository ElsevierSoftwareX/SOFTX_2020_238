#!/usr/bin/env python3
# Copyright (C) 2016  Aaron Viets
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
#				   Preamble
#
# =============================================================================
#


import numpy
import sys
from gstlal import pipeparts
from gstlal import calibration_parts
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_multiple_srcs(pipeline, name):

	#
	# This test uses multiple source elements
	#

	rate = 1000		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	src1 = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 32, verbose=False)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src1, "audio/x-raw, format=F32LE, rate=%d" % int(rate))
	src2 = test_common.test_src(pipeline, buffer_length = buffer_length, rate= rate, test_duration = test_duration, width=32, verbose = False)
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, src2, "audio/x-raw, format=F32LE, rate=%d" % int(rate))

	combined = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, capsfilter1, capsfilter2))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, combined), "multiple_srcs_out.dump")

	#
	# done
	#
	
	return pipeline
	
#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


test_common.build_and_run(lal_multiple_srcs, "lal_multiple_srcs")
