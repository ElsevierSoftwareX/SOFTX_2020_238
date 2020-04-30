#!/usr/bin/env python3
# Copyright (C) 2013  Kipp Cannon
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
# is the matrixmixer element an identity transform when given an identity
# matrix?
#


def matrixmixer_test_01(pipeline, name, width, channels):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	gap_frequency = 13.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	assert 1 <= channels <= 2
	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkmatrixmixer(pipeline, head, numpy.identity(channels, dtype = "double"))
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline


#
# test the transformation of a specific buffer with a specific matrix
#


def matrixmixer_test_02(name, dtype, samples, channels_in, channels_out):
	numpy.random.seed(0)
	input_array = numpy.random.random((samples, channels_in)).astype(dtype)
	# element always ingests mix matrix as double-precision floats
	mix = numpy.random.random((channels_in, channels_out)).astype("float64")
	# element will cast mix matrix to the appropriate type internally
	# for the matrix-matrix multiply
	output_reference = numpy.mat(input_array) * mix.astype(dtype)

	output_array, = test_common.transform_arrays([input_array], pipeparts.mkmatrixmixer, name, matrix = mix)

	if (output_array != output_reference).any():
		raise ValueError("incorrect output:  expected %s, got %s\ndifference = %s" % (output_reference, output_array, output_array - output_reference))


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(matrixmixer_test_01, "matrixmixer_test_01a", width = 64, channels = 1)
cmp_nxydumps.compare("matrixmixer_test_01a_in.dump", "matrixmixer_test_01a_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(matrixmixer_test_01, "matrixmixer_test_01b", width = 64, channels = 2)
cmp_nxydumps.compare("matrixmixer_test_01b_in.dump", "matrixmixer_test_01b_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(matrixmixer_test_01, "matrixmixer_test_01c", width = 32, channels = 1)
cmp_nxydumps.compare("matrixmixer_test_01c_in.dump", "matrixmixer_test_01c_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(matrixmixer_test_01, "matrixmixer_test_01d", width = 32, channels = 2)
cmp_nxydumps.compare("matrixmixer_test_01d_in.dump", "matrixmixer_test_01d_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)


matrixmixer_test_02("matrixmixer_test_02a", "float64", samples = 6, channels_in = 4, channels_out = 3)
matrixmixer_test_02("matrixmixer_test_02b", "float32", samples = 6, channels_in = 4, channels_out = 3)
