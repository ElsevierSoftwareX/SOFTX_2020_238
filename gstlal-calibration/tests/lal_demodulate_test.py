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

#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


test_common.build_and_run(lal_demodulate_01, "lal_demodulate_01")
#test_common.build_and_run(lal_demodulate_02, "lal_demodulate_02")
