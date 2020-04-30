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
import cmp_nxydumps
import test_common


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


def gate_test_01(pipeline, name):
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

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = 64, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	control = test_common.gapped_test_src(pipeline, buffer_length = buffer_length / 3, rate = rate, width = 32, test_duration = test_duration, gap_frequency = gap_frequency / 3, gap_threshold = gap_threshold, verbose = False)

	head = pipeparts.mkgate(pipeline, head, control = control, threshold = float("+inf"))

	pipeparts.mkfakesink(pipeline, head)

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


test_common.build_and_run(gate_test_01, "gate_test_01a")
