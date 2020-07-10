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

py_filters = numpy.load("/home/aaron.viets/src/gstlal/gstlal-calibration/tests/py_filter.npz")
mat_filters = numpy.load("/home/aaron.viets/src/gstlal/gstlal-calibration/tests/mat_filter.npz")

py_ctrl_corr_delay = py_filters["ctrl_corr_delay"]
py_ctrl_corr_filt = py_filters["ctrl_corr_filter"]

mat_ctrl_corr_delay = mat_filters["ctrl_corr_delay"]
mat_ctrl_corr_filt = mat_filters["ctrl_corr_filter"]


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def filters_01(pipeline, name):
	#
	# This tests whether the two filters files above behave identically when passed to a lal_firbank element. If so, the output file is ones.
	#

	rate = 32768		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 1000.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 64)
	tee = pipeparts.mktee(pipeline, src)
	py_filt = pipeparts.mkfirbank(pipeline, tee, latency = int(py_ctrl_corr_delay), fir_matrix = [py_ctrl_corr_filt[::-1]], time_domain = True)
	mat_filt = pipeparts.mkfirbank(pipeline, tee, latency = int(mat_ctrl_corr_delay), fir_matrix = [mat_ctrl_corr_filt[::-1]], time_domain = True)
	py_filt = pipeparts.mkaudiorate(pipeline, py_filt, skip_to_first = True, silent = False)
	mat_filt = pipeparts.mkaudiorate(pipeline, mat_filt, skip_to_first = True, silent = False)
	py_filt_inv = pipeparts.mkpow(pipeline, py_filt, exponent = -1.0)
	ratio = calibration_parts.mkmultiplier(pipeline, calibration_parts.list_srcs(pipeline, mat_filt, py_filt_inv))
	pipeparts.mknxydumpsink(pipeline, ratio, "%s_out.dump" % name)

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


test_common.build_and_run(filters_01, "filters_01")
