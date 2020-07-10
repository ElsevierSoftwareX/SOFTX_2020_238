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

def removeDC_01(pipeline, name):
	#
	# This test removes the DC component from a stream of ones (i.e., the result should be zero)
	#

	rate = 2048		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 50.0	# seconds
	DC = 1.0
	wave = 0
	volume = 0.0

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = 16384, test_duration = test_duration, wave = wave, width = 64)
	head = pipeparts.mkaudioamplify(pipeline, src, volume)
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = DC)
	head = calibration_parts.mkresample(pipeline, head, 5, True, "audio/x-raw,format=F64LE,rate=%d" % rate)
	head = calibration_parts.removeDC(pipeline, head, "audio/x-raw,format=F64LE,rate=%d" % rate)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

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


test_common.build_and_run(removeDC_01, "removeDC_01")

