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


#
# test the transformation of a specific buffer
#


def drop_test_02(name, dtype, length, drop_samples, sample_fuzz = cmp_nxydumps.default_sample_fuzz):
	channels_in = 1
	numpy.random.seed(0)
	# check that the first array is dropped
	input_array = numpy.random.random((length, channels_in)).astype(dtype)
	output_reference = input_array[drop_samples:]
	output_array = numpy.array(test_common.transform_arrays([input_array], pipeparts.mkdrop, name, drop_samples = drop_samples))
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


drop_test_02("drop_test_02a", "float64", length = 13147, drop_samples = 1337, sample_fuzz = cmp_nxydumps.default_sample_fuzz)
drop_test_02("drop_test_02a", "float32", length = 13147, drop_samples = 1337, sample_fuzz = cmp_nxydumps.default_sample_fuzz)
