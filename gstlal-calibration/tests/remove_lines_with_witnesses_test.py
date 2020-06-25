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
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def remove_lines_with_witnesses_test_01(pipeline, name):
	#
	# This tests generates a fake signal and fake witness channels
	# to test the removal of lines using the witnesses.
	#

	signal_rate = 1024	# Hz
	witness_rate = 1024	# Hz
	width = 64
	buffer_length = 1.0	# seconds
	test_duration = 500.0	# seconds

	compute_rate = 16	# Hz
	f0 = 60
	num_harmonics = 2
	f0_var = 0.02
	filter_latency = 1.0
	use_gate = False

	signal = test_common.test_src(pipeline, buffer_length = buffer_length, rate = signal_rate, width = width, channels = 1, test_duration = test_duration, wave = 5, freq = 0.1, volume = 1, src_suffix = "signal")

	witness0 = test_common.test_src(pipeline, buffer_length = buffer_length, rate = witness_rate, width = width, channels = 1, test_duration = test_duration, wave = 5, freq = 0.1, volume = 1, src_suffix = "witness0")
	witness1 = test_common.test_src(pipeline, buffer_length = buffer_length, rate = witness_rate, width = width, channels = 1, test_duration = test_duration, wave = 5, freq = 0.1, volume = 1, src_suffix = "witness1")

	witnesses = [witness0, witness1]

	noisesub_gate_bit = None
	if use_gate:
		signal = pipeparts.mktee(pipeline, signal)
		noisesub_gate_bit = calibration_parts.mkresample(pipeline, noisesub_gate_bit, 0, False, compute_rate)
		noisesub_gate_bit = pipeparts.mkgeneric(pipeline, noisesub_gate_bit, "lal_add_constant", value = 3.0)
		noisesub_gate_bit = pipeparts.mkgeneric(pipeline, noisesub_gate_bit, "lal_typecast")
		noisesub_gate_bit = pipeparts.mkcapsfilter(pipeline, noisesub_gate_bit, "audio/x-raw, format=U32LE, rate=%d" % compute_rate)

	clean = calibration_parts.remove_harmonics_with_witnesses(pipeline, signal, witnesses, f0, num_harmonics, f0_var, filter_latency, compute_rate = compute_rate, rate_out = signal_rate, num_median = 2048, num_avg = 160, noisesub_gate_bit = noisesub_gate_bit)
	pipeparts.mknxydumpsink(pipeline, clean, "clean.txt")


	#signal = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, signal, matrix = [[1.0,1.0]]))
	#witness0 = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, witness0, matrix = [[1.0,1.0]]))
	#witness1 = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, witness1, matrix = [[1.0,1.0]]))
	#allchan = [signal, witness0, witness1]
	#allchan = calibration_parts.mkinterleave(pipeline, allchan, complex_data = True)
	#allchan = pipeparts.mktee(pipeline, allchan)
	#pipeparts.mknxydumpsink(pipeline, allchan, "allchan.txt")
	#allchan = calibration_parts.mkdeinterleave(pipeline, allchan, 3, complex_data = True)
	#allchan = calibration_parts.mkadder(pipeline, allchan)
	#pipeparts.mknxydumpsink(pipeline, allchan, "allchan_sum.txt")

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


test_common.build_and_run(remove_lines_with_witnesses_test_01, "remove_lines_with_witnesses_test_01")


