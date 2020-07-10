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
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_transferfunction_01(pipeline, name):

	#
	# This test adds various noise into a stream and uses lal_transferfunction to remove it
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 3000.0	# seconds
	width = 64		# bits

	#
	# build pipeline
	#

	hoft = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 0.001, rate = rate, test_duration = test_duration, width = width, verbose = False)
	hoft = pipeparts.mktee(pipeline, hoft)

	common_noise = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, freq = 512, volume = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	common_noise = pipeparts.mktee(pipeline, common_noise)

	noise1 = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, freq = 512, volume = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	noise1 = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, common_noise, noise1))
	noise1 = pipeparts.mktee(pipeline, noise1)
	noise1_for_cleaning = pipeparts.mkgeneric(pipeline, noise1, "identity")
	hoft_noise1 = pipeparts.mkshift(pipeline, noise1, shift = 00000000)
	hoft_noise1 = pipeparts.mkaudioamplify(pipeline, hoft_noise1, 2)
	hoft_noise1 = pipeparts.mktee(pipeline, hoft_noise1)


	noise2 = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, freq = 1024, volume = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	noise2 = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, common_noise, noise2))
	noise2 = pipeparts.mktee(pipeline, noise2)
	noise2_for_cleaning = pipeparts.mkgeneric(pipeline, noise2, "identity")
	hoft_noise2 = pipeparts.mkshift(pipeline, noise2, shift = 00000)
	hoft_noise2 = pipeparts.mkaudioamplify(pipeline, hoft_noise2, 3)
	hoft_noise2 = pipeparts.mktee(pipeline, hoft_noise2)

	noisy_hoft = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, hoft, hoft_noise1, hoft_noise2))
	noisy_hoft = pipeparts.mktee(pipeline, noisy_hoft)

	clean_hoft = calibration_parts.clean_data(pipeline, calibration_parts.list_srcs(pipeline, noisy_hoft, noise1_for_cleaning, noise2_for_cleaning), rate / 4, rate / 8, 16384, test_duration * rate)
	clean_hoft = pipeparts.mktee(pipeline, clean_hoft)

#	hoft_inv = pipeparts.mkpow(pipeline, hoft, exponent = -1.0)
#	clean_hoft_over_hoft = calibration_parts.mkmultiplier(pipeline, calibration_parts.list_srcs(pipeline, hoft_inv, clean_hoft))
#	pipeparts.mknxydumpsink(pipeline, clean_hoft_over_hoft, "%s_clean_hoft_over_hoft.txt" % name)

	pipeparts.mknxydumpsink(pipeline, hoft, "%s_hoft.txt" % name)
	pipeparts.mknxydumpsink(pipeline, hoft_noise1, "%s_hoft_noise1.txt" % name)
	pipeparts.mknxydumpsink(pipeline, hoft_noise2, "%s_hoft_noise2.txt" % name)
	pipeparts.mknxydumpsink(pipeline, noisy_hoft, "%s_noisy_hoft.txt" % name)
	pipeparts.mknxydumpsink(pipeline, clean_hoft, "%s_clean_hoft.txt" % name)	

	#
	# done
	#
	
	return pipeline

def lal_transferfunction_02(pipeline, name):

	#
	# This test produces three-channel data to be read into lal_transferfunction
	#

	rate = 16384	    	# Hz
	buffer_length = 1.0	# seconds
	test_duration = 500.0	# seconds
	width = 64		# bits
	channels = 1
	freq = 512		# Hz

	#
	# build pipeline
	#

	hoft = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, freq = freq, channels = channels, rate = rate, test_duration = test_duration, width = width, verbose = False)
	hoft = pipeparts.mktee(pipeline, hoft)

	noise = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, freq = freq, channels = channels, rate = rate, test_duration = test_duration, width = width, verbose = False)
	noise = pipeparts.mktee(pipeline, noise)

	witness1 = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, calibration_parts.highpass(pipeline, hoft, rate, fcut = 400), calibration_parts.lowpass(pipeline, noise, rate, fcut = 400)))
	witness2 = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, calibration_parts.lowpass(pipeline, hoft, rate, fcut = 600), calibration_parts.highpass(pipeline, noise, rate, fcut = 600)))

	hoft = calibration_parts.highpass(pipeline, hoft, rate)
	clean_data = calibration_parts.clean_data(pipeline, hoft, rate, calibration_parts.list_srcs(pipeline, witness1, witness2), rate, 4 * rate, 2 * rate, 128, rate * test_duration, int(0.5 * rate), 10.0, filename = "highpass_lowpass_tfs.txt")
	pipeparts.mknxydumpsink(pipeline, clean_data, "%s_out.txt" % name)

	return pipeline

def lal_transferfunction_03(pipeline, name):

	#
	# This test produces three-channel data to be read into lal_transferfunction
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 50.0	# seconds
	width = 64		# bits
	freq = 512		# Hz
	num_witnesses = 2
	witness_factor = 0.5	# Ratio of amplitudes of witness / signal

	#
	# build pipeline
	#

	hoft = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, freq = freq, channels = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	hoft = pipeparts.mktee(pipeline, hoft)

	noise = []
	for i in range(0, num_witnesses):
		difference = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 0.001, channels = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
		noisy_hoft = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, hoft, witness_factor), difference))
		noisy_hoft = pipeparts.mkprogressreport(pipeline, noisy_hoft, "noisy_hoft_%d" % i)
		noise.append(noisy_hoft)

	clean_data = calibration_parts.clean_data(pipeline, hoft, rate, noise, rate, 2 * rate, rate, 10, rate * 1000, rate, 4, filename = "tfs.txt")
	pipeparts.mknxydumpsink(pipeline, hoft, "%s_hoft.txt" % name)
	pipeparts.mknxydumpsink(pipeline, clean_data, "%s_out.txt" % name)

	return pipeline

def lal_transferfunction_04(pipeline, name):

	#
	# This test produces three-channel data to be read into lal_transferfunction
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 500.0	# seconds
	width = 64		# bits
	channels = 1
	freq = 512		# Hz

	#
	# build pipeline
	#

	hoft = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, freq = freq, channels = channels, rate = rate, test_duration = test_duration, width = width, verbose = False)
	hoft = pipeparts.mktee(pipeline, hoft)
	shifted_hoft = pipeparts.mkgeneric(pipeline, hoft, "lal_shift", shift = 35024)
	interleaved_data = calibration_parts.mkinterleave(pipeline, calibration_parts.list_srcs(pipeline, hoft, shifted_hoft))
	pipeparts.mkgeneric(pipeline, interleaved_data, "lal_transferfunction", fft_length = 4 * rate, fft_overlap = 2 * rate, num_ffts = 64, update_samples = rate * test_duration, make_fir_filters = 1, fir_length = rate, update_after_gap = True, filename = "lpfilter.txt")

	return pipeline

#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


#test_common.build_and_run(lal_transferfunction_01, "lal_transferfunction_01")
#test_common.build_and_run(lal_transferfunction_02, "lal_transferfunction_02")
test_common.build_and_run(lal_transferfunction_03, "lal_transferfunction_03")
#test_common.build_and_run(lal_transferfunction_04, "lal_transferfunction_04")





