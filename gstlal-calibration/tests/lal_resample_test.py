#!/usr/bin/env python
# Copyright (C) 2017  Aaron Viets
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
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_resample_01(pipeline, name):

	#
	# This test passes an impulse through the resampler
	#

	rate_in = 128		# Hz
	rate_out = 1024		# Hz
	buffer_length = 10.0	# seconds
	test_duration = 30.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0.25, rate = rate_in, test_duration = test_duration, width = 64)
	head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [0.999999999, 1.00000001], block_duration = 0.5 * Gst.SECOND)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, 5, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#
	
	return pipeline


def lal_resample_02(pipeline, name):

	#
	# This test passes a sinusoid through the resampler
	#

	rate_in = 8192		# Hz
	rate_out = 16384	# Hz
	buffer_length = 1	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 1, rate = rate_in, test_duration = test_duration, width = 64)
	#head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-2, 2])
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, 5, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#

	return pipeline

def lal_resample_03(pipeline, name):

	#
	# This test passes ones through the resampler
	#

	rate_in = 128		# Hz
	rate_out = 1024		# Hz
	buffer_length = 0.25	# seconds
	test_duration = 30.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0.0, rate = rate_in, test_duration = test_duration, width = 64)
	#head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-2, 2])
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = 1)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, 5, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
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


test_common.build_and_run(lal_resample_01, "lal_resample_01")
test_common.build_and_run(lal_resample_02, "lal_resample_02")
test_common.build_and_run(lal_resample_03, "lal_resample_03")





