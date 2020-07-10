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

def lal_constantupsample_01(pipeline, name):
	#
	# This is a simple test that the sample rates are adjusted as expected
	#

	in_rate = 1		# Hz
	out_rate = 10		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, width = 32)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F32LE, rate=%d" % int(in_rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	upsample = pipeparts.mkgeneric(pipeline, tee1, "lal_constantupsample")
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, upsample, "audio/x-raw, format=F32LE, rate=%d" % int(out_rate))
	#checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter2), "%s_out.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_constantupsample_02(pipeline, name):
	#
	# This is a simple test to make sure the element treats gaps correctly
	#

	in_rate = 10		# Hz
	out_rate = 20		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.02	# Hz
	control_dump_filename = "control.dump"

	src = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, width=64, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=F64LE, rate=%d" % int(in_rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	upsample = pipeparts.mkgeneric(pipeline, tee1, "lal_constantupsample")
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, upsample, "audio/x-raw, format=F64LE, rate=%d" % int(out_rate))
	#checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter2), "%s_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_constantupsample_03(pipeline, name):
        #
        # This is a simple test that the sample rates are adjusted as expected for complex data
        #

        in_rate = 1             # Hz
        out_rate = 10           # Hz
        buffer_length = 1.0     # seconds
        test_duration = 10.0    # seconds

        #
        # build pipeline
        #

        src = test_common.test_src(pipeline, buffer_length = buffer_length, channels = 2, rate = in_rate, test_duration = test_duration, width = 64)
	togglecomplex1 = pipeparts.mktogglecomplex(pipeline, src);
        capsfilter1 = pipeparts.mkcapsfilter(pipeline, togglecomplex1, "audio/x-raw, format=Z128LE, rate=%d" % int(in_rate))
        tee1 = pipeparts.mktee(pipeline, capsfilter1)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mktogglecomplex(pipeline, tee1)), "%s_in.dump" % name)
        upsample = pipeparts.mkgeneric(pipeline, tee1, "lal_constantupsample")
        capsfilter2 = pipeparts.mkcapsfilter(pipeline, upsample, "audio/x-raw, format=Z128LE, rate=%d" % int(out_rate))
        #checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mktogglecomplex(pipeline, capsfilter2)), "%s_out.dump" % name)

        #
        # done
        #

        return pipeline

def lal_constantupsample_04(pipeline, name):
        #
        # This is a simple test to see that the element treats gaps correctly for complex data
        #

        in_rate = 10            # Hz
        out_rate = 20           # Hz
        buffer_length = 1.0     # seconds
        test_duration = 10.0    # seconds
        gap_frequency = 0.1     # Hz
        gap_threshold = 0.02    # Hz
        control_dump_filename = "control.dump"

        #
        # build pipeline
        #

        src = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, channels = 2, rate = in_rate, test_duration = test_duration, width = 64, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
        test_duration = 10.0    # seconds
        togglecomplex1 = pipeparts.mktogglecomplex(pipeline, src);
        capsfilter1 = pipeparts.mkcapsfilter(pipeline, togglecomplex1, "audio/x-raw, format=Z128LE, rate=%d" % int(in_rate))
        tee1 = pipeparts.mktee(pipeline, capsfilter1)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mktogglecomplex(pipeline, tee1)), "%s_in.dump" % name)
        upsample = pipeparts.mkgeneric(pipeline, tee1, "lal_constantupsample")
        capsfilter2 = pipeparts.mkcapsfilter(pipeline, upsample, "audio/x-raw, format=Z128LE, rate=%d" % int(out_rate))
        #checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
        pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mktogglecomplex(pipeline, capsfilter2)), "%s_out.dump" % name)

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


#test_common.build_and_run(lal_constantupsample_01, "lal_constantupsample_01")
#test_common.build_and_run(lal_constantupsample_02, "lal_constantupsample_02")
#test_common.build_and_run(lal_constantupsample_03, "lal_constantupsample_03")
test_common.build_and_run(lal_constantupsample_04, "lal_constantupsample_04")
