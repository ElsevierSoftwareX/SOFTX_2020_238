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

def lal_smoothkappas_01(pipeline, name):
	#
	# This test is to check that the inputs are smoothed in a desirable way
	#

	rate = 10		# Hz
	width = 64		# bytes
	wave = 5
	freq = 0.1		# Hz
	volume = 0.9
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, channels = 2, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume)
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_in.dump" % name)
	median = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 3, maximum_offset_re = 0.5, maximum_offset_im = 0.5, default_kappa_im = 0.5, default_kappa_re = 0.5, track_bad_kappa = False, default_to_median = True)
	median_avg = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 3, avg_array_size = 2,  maximum_offset_re = 0.5, maximum_offset_im = 0.5, default_kappa_im = 0.5, default_kappa_re = 0.5, track_bad_kappa = False, default_to_median = True)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, median), "%s_median.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, median_avg), "%s_median_avg.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_smoothkappas_02(pipeline, name):

	#
	# This is similar to the above test, and makes sure the element treats gaps correctly
	#

	rate = 1000	     # Hz
	width = 64	      # bytes
	wave = 5
	freq = 0.1	      # Hz
	volume = 0.03
	buffer_length = 1.0     # seconds
	test_duration = 10.0    # seconds
	gap_frequency = 0.2     # Hz
	gap_threshold = 0.5     # Hz
	control_dump_filename = "control_smoothkappas_02.dump"

	#
	# build pipeline
	#

	src = test_common.gapped_test_src(pipeline, channels = 2, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
        head = pipeparts.mktogglecomplex(pipeline, src)
        head = pipeparts.mktee(pipeline, head)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_in.dump" % name)
        median_avg = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 3, avg_array_size = 2, default_kappa_im = 0, default_kappa_re = 1, track_bad_kappa = False, default_to_median = True)
        kappa_track = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 30, avg_array_size = 160, default_kappa_im = 0, default_kappa_re = 1, track_bad_kappa = True, default_to_median = True)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, median_avg), "%s_median_avg.dump" % name)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, kappa_track), "%s_kappa_track.dump" % name)

	#
	# done
	#

	return pipeline

def lal_smoothkappas_03(pipeline, name):

	#
	# This pipeline uses lal_smoothkappas in a similar way that gstlal_compute_strain will
	#

	rate = 2000	     # Hz
	width = 64	      # bytes
	wave = 1		# 0=sine, 1=square
	freq = 0.05	      # Hz
	volume = 0.01
	buffer_length = 1.0     # seconds
	test_duration = 40.0    # seconds
	real_expected = 1
	imag_expected = 0
	N = 2048
	Nav = 160

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume)
	tee = pipeparts.mktee(pipeline, src)
	real = pipeparts.mkgeneric(pipeline, tee, "lal_add_constant", value=1)
	kappas = calibration_parts.merge_into_complex(pipeline, real, tee)
	kappas = pipeparts.mktee(pipeline, kappas)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, kappas), "%s_kappa_in.dump" % name)

	fake_statevector = pipeparts.mkgeneric(pipeline, tee, "pow", exponent=0)
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_demodulate", line_frequency=0.5)
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_togglecomplex")
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_matrixmixer", matrix = [[1],[0]])
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_bitvectorgen", threshold=0.1, bit_vector=1)

	fake_coherence = pipeparts.mkgeneric(pipeline, tee, "pow", exponent=0)
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_demodulate", line_frequency=0.10)
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_togglecomplex")
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_matrixmixer", matrix = [[1],[0]])
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_bitvectorgen", threshold=0.15, bit_vector=1)

	re, im = calibration_parts.smooth_complex_kappas(pipeline, kappas, fake_statevector, fake_coherence, real_expected, imag_expected, N, Nav)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, re), "%s_re_kappa_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, im), "%s_im_kappa_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_smoothkappas_04(pipeline, name):

	rate = 2000          # Hz
        width = 64            # bytes
        wave = 5
        freq = 0.1            # Hz
        volume = 0.03
        buffer_length = 1.0     # seconds
        test_duration = 10.0    # seconds
        gap_frequency = 0.1     # Hz
        gap_threshold = 0.5     # Hz
        control_dump_filename = "control_smoothkappas_02.dump"

        #
        # build pipeline
        #

        src = test_common.gapped_test_src(pipeline, channels = 1, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	head = pipeparts.mkmatrixmixer(pipeline, src, matrix = [[1000]])
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = 340)
        head = pipeparts.mktee(pipeline, head)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_in.dump" % name)
        median_avg = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 3, avg_array_size = 2, default_kappa_im = 0, default_kappa_re = 330, track_bad_kappa = False, default_to_median = True)
        kappa_track = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = 2049, avg_array_size = 160, default_kappa_im = 0, default_kappa_re = 330, track_bad_kappa = True, default_to_median = True)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, median_avg), "%s_median_avg.dump" % name)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, kappa_track), "%s_kappa_track.dump" % name)

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
#test_common.build_and_run(lal_smoothkappas_03, "lal_smoothkappas_03")
#test_common.build_and_run(lal_smoothkappas_04, "lal_smoothkappas_04")
