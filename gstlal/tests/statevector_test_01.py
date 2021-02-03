#!/usr/bin/env python3
# Copyright (C) 2014  Jolien Creighton
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




def statevector_test_01(name, width, samples):
	imax = 1 << width
	bits = 3		# number of bits to set for required-on and required-off
	required_on = 1
	required_off = 1

	while required_on & required_off:
		# random required-on and required-off bits
		required_on = int(sum(1 << bit for bit in numpy.random.randint(width - 1, size = bits))) & (imax - 1)
		required_off = int(sum(1 << bit for bit in numpy.random.randint(width - 1, size = bits))) & (imax - 1)

	input_samples = numpy.random.randint(imax, size=(samples, 1)).astype("u%d" % (width // 8))
	output_reference = ((input_samples & required_on) == required_on) & ((~input_samples & required_off) == required_off)
	output_array, = test_common.transform_arrays([input_samples], pipeparts.mkstatevector, name, required_on = required_on, required_off = required_off)
	output_array.dtype = bool
	if (output_array != output_reference).any():
		raise ValueError("incorrect output:  expected %s, got %s" % (output_reference, output_array))


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


for _ in range(100):
	statevector_test_01("statevector_test_01a", 32, samples = 1000)
