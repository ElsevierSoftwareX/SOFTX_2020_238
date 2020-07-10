#!/usr/bin/env python3
# Copyright (C) 2013  Madeline Wade
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
from gstlal import calibration_parts
import test_common_old

#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#

def lal_interleave_01(pipeline, name):

	caps = "audio/x-raw-float, width=64, rate=2048, channels=1"

	#
	# build pipeline
	#

	head = test_common_old.test_src(pipeline)
	headtee = pipeparts.mktee(pipeline, head)

	head1 = pipeparts.mkfirbank(pipeline, headtee, latency=-10, fir_matrix = [[0,1]])
	head1tee = pipeparts.mktee(pipeline, head1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head1tee), "%s_in_shifted.dump" % name)

	out = pipeparts.mkgeneric(pipeline, None, "lal_interleave", sync = True)
	#out = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync = True)
	pipeparts.mkqueue(pipeline, headtee).link(out)
	pipeparts.mkqueue(pipeline, head1tee).link(out)
	#out = calibration_parts.mkinterleave(pipeline, calibration_parts.list_srcs(pipeline, headtee, head1tee), caps)
	
	pipeparts.mknxydumpsink(pipeline, out, "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, headtee), "%s_in_notshifted.dump" % name)

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


test_common_old.build_and_run(lal_interleave_01, "lal_interleave_01")

