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

def lal_demodulate_01(pipeline, name):
	#
	# This test is to check that the inputs are multiplied by exp(2*pi*i*f*t) using the correct timestamps
	#

	rate = 1000		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 32)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F32LE, rate=%d" % int(rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	demodulate = pipeparts.mkgeneric(pipeline, tee1, "lal_demodulate", line_frequency=100)
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, demodulate, "audio/x-raw, format=Z64LE, rate=%d" % int(rate))
	togglecomplex = pipeparts.mktogglecomplex(pipeline, capsfilter2)
	capsfilter3 = pipeparts.mkcapsfilter(pipeline, togglecomplex, "audio/x-raw, format=F32LE, rate=%d" % int(rate))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter3), "%s_out.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_demodulate_02(pipeline, name):
	#
	# This is similar to the above test, and makes sure the element treats gaps correctly
	#

	rate = 1000		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.2	# Hz
	control_dump_filename = "control_demodulate_02.dump"

	src = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width=64, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F64LE, rate=%d" % int(rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	demodulate = pipeparts.mkgeneric(pipeline, tee1, "lal_demodulate")
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, demodulate, "audio/x-raw, format=Z128LE, rate=%d" % int(rate))
	togglecomplex = pipeparts.mktogglecomplex(pipeline, capsfilter2)
	capsfilter3 = pipeparts.mkcapsfilter(pipeline, togglecomplex, "audio/x-raw, format=F64LE, rate=%d" %int(rate))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter3), "%s_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_demodulate_03(pipeline, name):
	#
	# This test checks sensitivity of the demodulation process used in the calibration pipeline to small changes in line frequency
	#

	rate_in = 16384	    	# Hz
	rate_out = 16		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 1000	# seconds

	#
	# build pipeline
	#

	# Make fake data with a signal
	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate_in, test_duration = test_duration, wave = 0, volume = 1.0, freq = 37.00, width = 64)

	# Demodulate it
	head = calibration_parts.demodulate(pipeline, src, 37.10, True, rate_out, 20, 0.0)

	# Smoothing
#	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 128 * rate_out, avg_array_size = 10 * rate_out)

	# Measure the amplitude of the result
	head = pipeparts.mkgeneric(pipeline, head, "cabs")
	head = pipeparts.mkaudioamplify(pipeline, head, 2.0)
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


#test_common.build_and_run(lal_demodulate_01, "lal_demodulate_01")
#test_common.build_and_run(lal_demodulate_02, "lal_demodulate_02")
test_common.build_and_run(lal_demodulate_03, "lal_demodulate_03")

