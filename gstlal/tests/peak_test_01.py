#!/usr/bin/env python3
# Copyright (C) 2014  Kipp Cannon
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
import cmp_nxydumps


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


#
# is the element an identity transform when given 1 channel of 1s and n = 1 ?
#


def peak_test_01(pipeline, name, width):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	gap_frequency = 13.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds
	n = 1			# samples

	#
	# build pipeline.  square wave with 0 frequency = stream of 1s
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, wave = 1, freq = 0, channels = 1, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkpeak(pipeline, head, n = n)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline


#
# test the transformation of a specific buffer
#


def peak_test_02(name, dtype, arrays, n, sample_fuzz = cmp_nxydumps.default_sample_fuzz):
	channels_in = 1
	numpy.random.seed(0)
	input_array = None
	output_reference = None
	for a in range(arrays):
		IA = numpy.random.random((n, channels_in)).astype(dtype)
		OR = numpy.copy(IA)
		OR[OR < max(OR)] = 0.
		if input_array is not None:
			input_array = numpy.append(input_array, IA, axis = 0)
			output_reference = numpy.append(output_reference, OR, axis = 0)
		else:
			input_array = IA
			output_reference = OR	

	output_array = numpy.reshape(numpy.ndarray.flatten(numpy.array(test_common.transform_arrays([input_array], pipeparts.mkpeak, name, n = n))), output_reference.shape)
	residual = abs((output_array - output_reference))
	if residual[residual > sample_fuzz].any():
		raise ValueError("incorrect output:  expected %s, got %s\ndifference = %s" % (output_reference, output_array, residual))


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(peak_test_01, "peak_test_01a", width = 64)
cmp_nxydumps.compare("peak_test_01a_in.dump", "peak_test_01a_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(peak_test_01, "peak_test_01b", width = 32)
cmp_nxydumps.compare("peak_test_01b_in.dump", "peak_test_01b_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)

peak_test_02("peak_test_02a", "float64", arrays = 29, n = 117, sample_fuzz = cmp_nxydumps.default_sample_fuzz)
peak_test_02("peak_test_02b", "float32", arrays = 29, n = 117, sample_fuzz = cmp_nxydumps.default_sample_fuzz**.5)
