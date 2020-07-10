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

def pyfilter_test_01(pipeline, name):
	#
	# This test removes the DC component from a stream of ones (i.e., the result should be zero)
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	DC = 1.0
	wave = 0
	freq = 90
	volume = 1.0

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, freq = freq, test_duration = test_duration, wave = wave, width = 64)
	head = pipeparts.mkaudioamplify(pipeline, src, volume)
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = DC)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = calibration_parts.bandstop(pipeline, head, rate)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

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


test_common.build_and_run(pyfilter_test_01, "pyfilter_test_01")

