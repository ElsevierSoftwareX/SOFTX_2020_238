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



def togglecomplex_test_01(pipeline, name, width, channels):
	#
	# try changing these.  test should still work!
	#

	initial_channels = 2	# number of channels to generate
	rate = 2048	# Hz
	gap_frequency = 13.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	mix = numpy.random.random((initial_channels, channels)).astype("float64")

	#
	# build pipeline
	#

	assert 1 <= initial_channels <= 2
	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = initial_channels, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix = mix)
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mkchecktimestamps(pipeline, head)

	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline



#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(togglecomplex_test_01, "togglecomplex_test_01a", width = 64, channels = 2)
cmp_nxydumps.compare("togglecomplex_test_01a_in.dump", "togglecomplex_test_01a_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(togglecomplex_test_01, "togglecomplex_test_01b", width = 64, channels = 4)
cmp_nxydumps.compare("togglecomplex_test_01b_in.dump", "togglecomplex_test_01b_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(togglecomplex_test_01, "togglecomplex_test_01c", width = 32, channels = 2)
cmp_nxydumps.compare("togglecomplex_test_01c_in.dump", "togglecomplex_test_01c_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
test_common.build_and_run(togglecomplex_test_01, "togglecomplex_test_01d", width = 32, channels = 4)
cmp_nxydumps.compare("togglecomplex_test_01d_in.dump", "togglecomplex_test_01d_out.dump", flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS)
