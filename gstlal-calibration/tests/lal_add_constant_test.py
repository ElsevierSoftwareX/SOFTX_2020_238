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
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_add_constant_01(pipeline, name):

	#
	# This test adds a constant to a stream of single-precision floats
	#

	rate = 1000		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 32)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F32LE, rate=%d" % int(rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	add_constant = pipeparts.mkgeneric(pipeline, tee1, "lal_add_constant", value=3)
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, add_constant, "audio/x-raw, format=F32LE, rate=%d" % int(rate))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter2), "%s_out.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_add_constant_02(pipeline, name):
        
        #
        # This test adds a constant to a stream of double-precision floats
        #

        rate = 1000             # Hz
        buffer_length = 1.0     # seconds
        test_duration = 10.0    # seconds

        #
        # build pipeline
        #

        src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 64)
        capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F64LE, rate=%d" % int(rate))
        tee1 = pipeparts.mktee(pipeline, capsfilter1)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
        add_constant = pipeparts.mkgeneric(pipeline, tee1, "lal_add_constant", value=3)
        capsfilter2 = pipeparts.mkcapsfilter(pipeline, add_constant, "audio/x-raw, format=F64LE, rate=%d" % int(rate))
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter2), "%s_out.dump" % name)

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


#test_common.build_and_run(lal_add_constant_01, "lal_add_constant_01")
test_common.build_and_run(lal_add_constant_02, "lal_add_constant_02")
