#!/usr/bin/env python3
# Copyright (C) 2017  Aaron Viets
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
#				  Utilities
#
# =============================================================================
#


def fir_matrix_update(elem, arg, filtered):
	filtered.set_property("kernel", elem.get_property("fir_matrix")[0][::-1])
	print("fir matrix updated")


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_fcc_update_01(pipeline, name):

	#
	# This test passes an impulse through the fcc_updater
	#

	data_rate = 16384		# Hz
	fcc_rate = 16			# Hz
	fcc_default = 360		# Hz
	fcc_update = 345		# Hz
	fcc_averaging_time = 5		# seconds
	fcc_filter_duration = 1		# seconds
	fcc_filter_taper_length = 32768	# seconds
	impulse_separation = 1.0	# seconds
	buffer_length = 1.0		# seconds
	test_duration = 10.0		# seconds

	#
	# build pipeline
	#

	src = pipeparts.mktee(pipeline, test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 1.0 / impulse_separation, rate = data_rate, test_duration = test_duration, width = 64))

	impulses = calibration_parts.mkinsertgap(pipeline, src, bad_data_intervals = [0.999999999, 1.00000001], block_duration = buffer_length * Gst.SECOND)
	impulses = pipeparts.mktee(pipeline, impulses)
	pipeparts.mknxydumpsink(pipeline, impulses, "%s_impulses.txt" % name)

	fcc = calibration_parts.mkresample(pipeline, src, 0, False, "audio/x-raw,format=F64LE,rate=%d" % fcc_rate)
	fcc = pipeparts.mkpow(pipeline, fcc, exponent = 0.0)
	fcc = pipeparts.mkgeneric(pipeline, fcc, "lal_add_constant", value = fcc_update - 1)
	fcc = pipeparts.mktee(pipeline, fcc)
	pipeparts.mknxydumpsink(pipeline, fcc, "%s_fcc.txt" % name)
	update_fcc = pipeparts.mkgeneric(pipeline, fcc, "lal_fcc_update", data_rate = data_rate, fcc_rate = fcc_rate, fcc_model = fcc_default, averaging_time = fcc_averaging_time, filter_duration = fcc_filter_duration)
	pipeparts.mkfakesink(pipeline, update_fcc)


	default_fir_matrix = numpy.zeros(int(numpy.floor(data_rate * fcc_filter_duration / 2.0 + 1) * 2.0 - 2.0))
	latency = int(data_rate * fcc_filter_duration / 2.0 + 1)
	default_fir_matrix[latency] = 1.0
	res = pipeparts.mkgeneric(pipeline, impulses, "lal_tdwhiten", kernel = default_fir_matrix[::-1], latency = latency, taper_length = fcc_filter_taper_length)
	update_fcc.connect("notify::fir-matrix", fir_matrix_update, res)
	pipeparts.mknxydumpsink(pipeline, res, "%s_out.txt" % name)

	#
	# done
	#
	
	return pipeline


def lal_fcc_update_02(pipeline, name):

	#
	# This test passes a sinusoid through the fcc_updater
	#

	data_rate = 16384		# Hz
	data_frequency = 1024		# Hz
	fcc_rate = 16			# Hz
	fcc_default = 360		# Hz
	fcc_update = 345		# Hz
	fcc_averaging_time = 5		# seconds
	fcc_filter_duration = 1		# seconds
	fcc_filter_taper_length = 32768	# seconds
	buffer_length = 1.0		# seconds
	test_duration = 10.0		# seconds

	#
	# build pipeline
	#

	src = pipeparts.mktee(pipeline, test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = data_frequency, rate = data_rate, test_duration = test_duration, width = 64))

	sinusoid = pipeparts.mktee(pipeline, src)
	pipeparts.mknxydumpsink(pipeline, sinusoid, "%s_sinusoid.txt" % name)

	fcc = calibration_parts.mkresample(pipeline, src, 0, False, "audio/x-raw,format=F64LE,rate=%d" % fcc_rate)
	fcc = pipeparts.mkpow(pipeline, fcc, exponent = 0.0)
	fcc = pipeparts.mkgeneric(pipeline, fcc, "lal_add_constant", value = fcc_update - 1)
	fcc = pipeparts.mktee(pipeline, fcc)
	pipeparts.mknxydumpsink(pipeline, fcc, "%s_fcc.txt" % name)
	update_fcc = pipeparts.mkgeneric(pipeline, fcc, "lal_fcc_update", data_rate = data_rate, fcc_rate = fcc_rate, fcc_model = fcc_default, averaging_time = fcc_averaging_time, filter_duration = fcc_filter_duration)
	pipeparts.mkfakesink(pipeline, update_fcc)


	default_fir_matrix = numpy.zeros(int(numpy.floor(data_rate * fcc_filter_duration / 2.0 + 1) * 2.0 - 2.0))
	latency = int(data_rate * fcc_filter_duration / 2.0 + 1)
	default_fir_matrix[latency] = 1.0
	res = pipeparts.mkgeneric(pipeline, sinusoid, "lal_tdwhiten", kernel = default_fir_matrix[::-1], latency = latency, taper_length = fcc_filter_taper_length)
	update_fcc.connect("notify::fir-matrix", fir_matrix_update, res)
	pipeparts.mknxydumpsink(pipeline, res, "%s_out.txt" % name)

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


test_common.build_and_run(lal_fcc_update_01, "lal_fcc_update_01")
test_common.build_and_run(lal_fcc_update_02, "lal_fcc_update_02")




