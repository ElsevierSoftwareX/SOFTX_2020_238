#!/usr/bin/env python
# Copyright (C) 2009,2010  Kipp Cannon
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
#                                   Preamble
#
# =============================================================================
#


import numpy
import sys
from gstlal import pipeparts
import cmp_nxydumps
import test_common


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


#
# is the firbank element an identity transform when given a unit impulse?
# in and out timeseries should be identical modulo start/stop transients
#


def firbank_test_01(pipeline, name, width, time_domain):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	gap_frequency = 13.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	fir_length = 21	# samples
	latency = (fir_length - 1) / 2	# samples, in [0, fir_length)

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = tee = pipeparts.mktee(pipeline, head)

	fir_matrix = numpy.zeros((1, fir_length), dtype = "double")
	fir_matrix[0, (fir_matrix.shape[1] - 1) - latency] = 1.0

	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = fir_matrix, latency = latency, time_domain = time_domain)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(firbank_test_01, "firbank_test_01a", width = 64, time_domain = True)
test_common.build_and_run(firbank_test_01, "firbank_test_01b", width = 64, time_domain = False)
test_common.build_and_run(firbank_test_01, "firbank_test_01c", width = 32, time_domain = True)
test_common.build_and_run(firbank_test_01, "firbank_test_01d", width = 32, time_domain = False)

flags = cmp_nxydumps.COMPARE_FLAGS_EXACT_GAPS | cmp_nxydumps.COMPARE_FLAGS_ZERO_IS_GAP | cmp_nxydumps.COMPARE_FLAGS_ALLOW_STARTSTOP_MISALIGN

cmp_nxydumps.compare("firbank_test_01a_in.dump", "firbank_test_01a_out.dump", flags = flags)
cmp_nxydumps.compare("firbank_test_01b_in.dump", "firbank_test_01b_out.dump", flags = flags)
cmp_nxydumps.compare("firbank_test_01c_in.dump", "firbank_test_01c_out.dump", flags = flags, sample_fuzz = 1e-6)
cmp_nxydumps.compare("firbank_test_01d_in.dump", "firbank_test_01d_out.dump", flags = flags, sample_fuzz = 1e-6)
