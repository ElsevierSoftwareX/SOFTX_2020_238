#!/usr/bin/env python
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
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_smoothkappas_01(pipeline, name):
	#
	# This test is to check that the inputs are multiplied by exp(2*pi*i*f*t) using the correct timestamps
	#

	rate = 1024		# Hz
	width = 64		# bytes
	wave = 1
	freq = 0.1		# Hz
	volume = 0.03
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume)
	add_constant = pipeparts.mkgeneric(pipeline, src, "lal_add_constant", value=1)
	tee = pipeparts.mktee(pipeline, add_constant)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)
	smoothkappas = pipeparts.mkgeneric(pipeline, tee, "lal_smoothkappas")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, smoothkappas), "%s_out.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_smoothkappas_02(pipeline, name):

	#
	# This is similar to the above test, and makes sure the element treats gaps correctly
	#

        rate = 1000             # Hz
        width = 64              # bytes
        wave = 1
        freq = 0.1              # Hz
        volume = 0.03
        buffer_length = 1.0     # seconds
        test_duration = 10.0    # seconds
	gap_frequency = 1     # Hz
        gap_threshold = 0.5     # Hz
        control_dump_filename = "control_smoothkappas_02.dump"

        #
        # build pipeline
        #

        src = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
        add_constant = pipeparts.mkgeneric(pipeline, src, "lal_add_constant", value=1)
        tee = pipeparts.mktee(pipeline, add_constant)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)
        smoothkappas = pipeparts.mkgeneric(pipeline, tee, "lal_smoothkappas")
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, smoothkappas), "%s_out.dump" % name)

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


#test_common.build_and_run(lal_smoothkappas_01, "lal_smoothkappas_01")
test_common.build_and_run(lal_smoothkappas_02, "lal_smoothkappas_02")
