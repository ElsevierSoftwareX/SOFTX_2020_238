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
from gstlal import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def exact_kappas_01(pipeline, name):

	#
	# Make a bunch of fake data at 16 Hz to pass through the exact kappas function.
	#

	rate_in = 16		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 300.0	# seconds
	num_stages = 1		# stages of actuation

	freqs = [17.1, 410.1, 17.6]# 16.4, 15.6]
	EPICS = []
	for i in range(2 * (1 + num_stages) * (2 + num_stages)):
		EPICS.append(numpy.random.rand())

	#
	# build pipeline
	#

	X = []
	for i in range(2 + num_stages):
		X.append(test_common.complex_test_src(pipeline, buffer_length = buffer_length, rate = rate_in, width = 64, test_duration = test_duration, wave = 5, freq = 0, src_suffix = str(i)))
	kappas = calibration_parts.compute_exact_kappas_from_filters_file(pipeline, X, freqs, EPICS, rate_in)
	for i in range(4 + 2 * num_stages):
		pipeparts.mknxydumpsink(pipeline, kappas[i], "kappas_%d.txt" % i)

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


test_common.build_and_run(exact_kappas_01, "exact_kappas_01")


