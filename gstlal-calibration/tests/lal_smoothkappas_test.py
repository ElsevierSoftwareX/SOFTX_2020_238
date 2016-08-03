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

	rate = 1000	     # Hz
	width = 64	      # bytes
	wave = 1
	freq = 0.1	      # Hz
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

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume)
	tee = pipeparts.mktee(pipeline, src)
	kappa = pipeparts.mkgeneric(pipeline, tee, "lal_add_constant", value=1)
	kappa = pipeparts.mktee(pipeline, kappa)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, kappa), "%s_kappa_in.dump" % name)

	fake_statevector = pipeparts.mkgeneric(pipeline, tee, "pow", exponent=0)
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_demodulate", line_frequency=0.5)
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_togglecomplex")
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_matrixmixer", matrix = [[1],[0]])
	fake_statevector = pipeparts.mkgeneric(pipeline, fake_statevector, "lal_bitvectorgen", threshold=0.1, bit_vector=1)
	fake_statevector = pipeparts.mktee(pipeline, fake_statevector)

	fake_coherence = pipeparts.mkgeneric(pipeline, tee, "pow", exponent=0)
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_demodulate", line_frequency=0.10)
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_togglecomplex")
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_matrixmixer", matrix = [[1],[0]])
	fake_coherence = pipeparts.mkgeneric(pipeline, fake_coherence, "lal_bitvectorgen", threshold=0.15, bit_vector=1)

	to_kappa_or_not_to_kappa = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, fake_statevector, fake_coherence))
	gapped_kappas = pipeparts.mkgate(pipeline, calibration_parts.mkqueue(pipeline, kappa), control = calibration_parts.mkqueue(pipeline, to_kappa_or_not_to_kappa), threshold = 2)
	gapped_kappas = pipeparts.mkgeneric(pipeline, gapped_kappas, "audiorate")

	smooth_kappas = pipeparts.mkgeneric(pipeline, gapped_kappas, "lal_smoothkappas", kappa_ceiling=0.01)
	smooth_kappas = pipeparts.mkgate(pipeline, calibration_parts.mkqueue(pipeline, smooth_kappas), control = calibration_parts.mkqueue(pipeline, fake_statevector), threshold=1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, smooth_kappas), "%s_kappa_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_smoothkappas_04(pipeline, name):

	rate = 1000	  # Hz
	width = 64	    # bytes
	wave = 1		# 0=sine, 1=square
	freq = 0.1	    # Hz
	volume = 0.03
	buffer_length = 1.0     # seconds
	test_duration = 10.0    # seconds

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, volume = volume, channels = 2)
	matrixmixer = pipeparts.mkgeneric(pipeline, src, "lal_matrixmixer", matrix = [[1],[0]])
	sink = pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, matrixmixer), "%s_out.dump" % name)

	return pipeline

#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


#test_common.build_and_run(lal_smoothkappas_01, "lal_smoothkappas_01")
#test_common.build_and_run(lal_smoothkappas_02, "lal_smoothkappas_02")
test_common.build_and_run(lal_smoothkappas_03, "lal_smoothkappas_03")
#test_common.build_and_run(lal_smoothkappas_04, "lal_smoothkappas_04")
