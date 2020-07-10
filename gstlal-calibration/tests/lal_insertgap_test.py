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

def lal_insertgap_test_01(pipeline, name):
	#
	# This test should first replace negative numbers with zeros, and then replace them.
	#

	rate = 1000		# Hz
	width = 64
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.0
	control_dump_filename = "control_insertgap_test_01.dump"
	#bad_data_intervals = numpy.random.random((4,)).astype("float64")
	#bad_data_intervals2 = numpy.random.random((4,)).astype("float64")
	bad_data_intervals = [-1e35, -1.0, 0.0, 1e35]
	bad_data_intervals2 = [-1e35, 0.0, 0.0, 1e35]

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = 1, test_duration = test_duration, wave = 0, freq = 1, volume = 1, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	#head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = 1, test_duration = test_duration, wave = 5, freq = 0, volume = 1)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = False, fill_discont = True, replace_value = 0.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.dump" % name)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals2, insert_gap = True, fill_discont = True, replace_value = 7.0)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_insertgap_test_02(pipeline, name):
	#
	# This tests the element's treatment of complex data streams.
	#

	rate = 1000	     # Hz
	width = 32
	buffer_length = 1.0     # seconds
	test_duration = 100.0   # seconds
	gap_frequency = 0.1     # Hz
	gap_threshold = 0.0
	control_dump_filename = "control_insertgap_test_01.dump"
	#bad_data_intervals = numpy.random.random((4,)).astype("float64")
	#bad_data_intervals2 = numpy.random.random((4,)).astype("float64")
	bad_data_intervals = [-1e35, -1.0, -0.5, 1e35]
	bad_data_intervals2 = [-1e35, 0.5, 1.0, 1e35]

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = 2, test_duration = test_duration, wave = 0, freq = 1, volume = 1, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	head = pipeparts.mktogglecomplex(pipeline, head)
	#head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = 1, test_duration = test_duration, wave = 5, freq = 0, volume = 1)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = True, fill_discont = True, replace_value = 3.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.dump" % name)
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals2, insert_gap = True, remove_gap = True, fill_discont = True, replace_value = 7.0)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_insertgap_test_03(pipeline, name):
	#
	# This tests how the element handles 32 bit unsigned integers
	#

	rate = 1000	     # Hz
	width = 32
	buffer_length = 1.0     # seconds
	test_duration = 100.0   # seconds
	bad_data_intervals = numpy.random.randn(10)


	print("lal_insertgap_test_03 bad_data_intervals = %s" % bad_data_intervals)
	head = test_common.int_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration)
	head = pipeparts.mkaudioconvert(pipeline, head, "audio/x-raw,format=U32LE")
	head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = False, fill_discont = True, replace_value = 0.0)
	head = pipeparts.mktee(pipeline, head)
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


test_common.build_and_run(lal_insertgap_test_01, "lal_insertgap_test_01")
test_common.build_and_run(lal_insertgap_test_02, "lal_insertgap_test_02")
test_common.build_and_run(lal_insertgap_test_03, "lal_insertgap_test_03")


