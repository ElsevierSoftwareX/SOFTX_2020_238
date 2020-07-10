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


def lal_gate_01(pipeline, name):
	#
	# This pipeline tests the behavior of lal_gate with respect to attack_length, start-of-stream, etc.
	#

	rate = 512	    	# Hz
	buffer_length = 1.0	# seconds
	test_duration = 16	# seconds
	frequency = 0.1		# Hz
	attack_length = -3	# seconds
	hold_length = 0		# seconds
	threshold = 1.0
	DC_offset = 1.0

	#
	# build pipeline
	#

	# Make a sine wave
	src = test_common.complex_test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 0, freq = frequency, width = 64)

	# Add a DC offset
	head = pipeparts.mkgeneric(pipeline, src, "lal_add_constant", value = DC_offset)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	control = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 0, volume = 1.0, freq = frequency, width = 64, src_suffix = "_control")
	control = pipeparts.mkgeneric(pipeline, control, "lal_add_constant", value = DC_offset)
	#control = pipeparts.mkgeneric(pipeline, control, "splitcounter", name = "control")
	#head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = "before")

	# Gate it
	head = calibration_parts.mkgate(pipeline, head, control, threshold, attack_length = int(attack_length * rate), hold_length = int(hold_length * rate))
	#head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = "after")
	real, imag = calibration_parts.split_into_real(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, real, "%s_real_out.txt" % name)
	pipeparts.mknxydumpsink(pipeline, imag, "%s_imag_out.txt" % name)

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


test_common.build_and_run(lal_gate_01, "lal_gate_01")

