#!/usr/bin/env python
# Copyright (C) 2019  Aaron Viets
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
from gstlal import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_matrixsolver_01(pipeline, name):

	#
	# Make a bunch of fake data at 16 Hz to pass through the exact kappas function.
	#

	rate_in = 16		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 300.0	# seconds
	channels = 10 * 11	# inputs to lal_matrixsolver

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate_in, width = 64, channels = channels, test_duration = test_duration, wave = 5, freq = 0)
	streams = calibration_parts.mkdeinterleave(pipeline, head, channels)
	head = calibration_parts.mkinterleave(pipeline, streams)
	solutions = pipeparts.mkgeneric(pipeline, head, "lal_matrixsolver")
	solutions = list(calibration_parts.mkdeinterleave(pipeline, solutions, 10))
	for i in range(10):
		pipeparts.mknxydumpsink(pipeline, solutions[i], "solutions_%d.txt" % i)

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


test_common.build_and_run(lal_matrixsolver_01, "lal_matrixsolver_01")


