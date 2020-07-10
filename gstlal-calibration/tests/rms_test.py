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
from gstlal import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def rms_01(pipeline, name):

	#
	# This test passes a random series of integers through rms
	#

	rate_in = 2048		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 50.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate_in, width = 64, test_duration = test_duration, wave = 5, freq = 0)
	white_noise = pipeparts.mktee(pipeline, head)
	white_noise_over10 = pipeparts.mkaudioamplify(pipeline, white_noise, 0.1)
	white_noise_bp100to150 = calibration_parts.bandpass(pipeline, white_noise, 2048, length = 1.0, f_low = 100, f_high = 150)
	white_noise_bp250to300 = calibration_parts.bandpass(pipeline, white_noise, 2048, length = 1.0, f_low = 250, f_high = 300)
	white_noise_bp450to500 = calibration_parts.bandpass(pipeline, white_noise, 2048, length = 1.0, f_low = 450, f_high = 500)
	white_noise_bp400to500 = calibration_parts.bandpass(pipeline, white_noise, 2048, length = 1.0, f_low = 400, f_high = 500)
	rms = calibration_parts.compute_rms(pipeline, white_noise, 1024, 1.0, f_min = 100, f_max = 500)
	rms_over10 = calibration_parts.compute_rms(pipeline, white_noise_over10, 1024, 1.0, f_min = 100, f_max = 500)
	rms_bp100to150 = calibration_parts.compute_rms(pipeline, white_noise_bp100to150, 1024, 1.0, f_min = 100, f_max = 500)
	rms_bp250to300 = calibration_parts.compute_rms(pipeline, white_noise_bp250to300, 1024, 1.0, f_min = 100, f_max = 500)
	rms_bp450to500 = calibration_parts.compute_rms(pipeline, white_noise_bp450to500, 1024, 1.0, f_min = 100, f_max = 500)
	rms_bp400to500 = calibration_parts.compute_rms(pipeline, white_noise_bp400to500, 1024, 1.0, f_min = 100, f_max = 500)
	pipeparts.mknxydumpsink(pipeline, rms, "rms.txt")
	pipeparts.mknxydumpsink(pipeline, rms_over10, "rms_over10.txt")
	pipeparts.mknxydumpsink(pipeline, rms_bp100to150, "rms_bp100to150.txt")
	pipeparts.mknxydumpsink(pipeline, rms_bp250to300, "rms_bp250to300.txt")
	pipeparts.mknxydumpsink(pipeline, rms_bp450to500, "rms_bp450to500.txt")
	pipeparts.mknxydumpsink(pipeline, rms_bp400to500, "rms_bp400to500.txt")

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


test_common.build_and_run(rms_01, "rms_01")


