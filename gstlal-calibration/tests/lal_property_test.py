#!/usr/bin/env python3
# Copyright (C) 2018  Aaron Viets
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

def lal_property_test_01(pipeline, name):
	#
	# This test makes a stream changing between 2's and 8's.
	# It should square the 2's and take the 8's to the 8th power.
	#

	rate = 512		# Hz
	width = 64
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.0
	control_dump_filename = "control_property_test_01.dump"
	bad_data_intervals2 = [0.0, 1e35]
	bad_data_intervals = [-1e35, 1e-35]

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = 1, test_duration = test_duration, wave = 0, freq = 0.1, volume = 1)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = False, fill_discont = True, replace_value = 2.0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals2, insert_gap = False, fill_discont = True, replace_value = 8.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.dump" % name)
	lal_prop_exponent = pipeparts.mkgeneric(pipeline, head, "lal_property", update_when_change = True)
	head = calibration_parts.mkpow(pipeline, head, exponent = 0.0)

	lal_prop_exponent.connect("notify::current-average", calibration_parts.update_property_simple, head, "current_average", "exponent")

	pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

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


test_common.build_and_run(lal_property_test_01, "lal_property_test_01")


