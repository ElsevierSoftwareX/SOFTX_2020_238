#!/usr/bin/env python
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
import test_common

#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#

def complex_pow_01(pipeline, name):

	caps = "audio/x-raw-float, width=64, rate=2048, channels=1"

	#
	# build pipeline
	#

	src1 = pipeparts.mkaudiotestsrc(pipeline, None, wave=0, freq=1024)
	src2 = pipeparts.mkaudiotestsrc(pipeline, None, wave=1, freq=2048)

	src1 = pipeparts.mkqueue(pipeline, src1)
	src2 = pipeparts.mkqueue(pipeline, src2)


	out = pipeparts.mkgeneric(pipeline, None, "interleave")
	src1.link(out)
	src2.link(out)
	
	out = pipeparts.mktogglecomplex(pipeline, out)
	outtee = pipeparts.mktee(pipeline, out)
	pipeparts.mknxydumpsink(pipeline, outtee, "before_pow.dump")	

	out = pipeparts.mktogglecomplex(pipeline, outtee)
	out = pipeparts.mkgeneric(pipeline, out, "complex_pow", exponent=2)
	out = pipeparts.mktogglecomplex(pipeline, out) 


	pipeparts.mknxydumpsink(pipeline, out, "after_pow.dump")

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


test_common.build_and_run(complex_pow_01, "complex_pow_01")

