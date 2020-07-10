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
import test_common


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#

def lal_logical_undersampler_01(pipeline, name):
	#
	# This is a simple test that the sample rates are adjusted as expected
	#

	in_rate = 10		# Hz
	out_rate = 1		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src_ints(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration)
	head = tee = pipeparts.mktee(pipeline, head)
	
	head = pipeparts.mkgeneric(pipeline, tee, "lal_logical_undersampler", required_on = 0x1, status_out = 0x7)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-int, rate=%d" % int(out_rate))
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#
	
	return pipeline
	
def lal_logical_undersampler_02(pipeline, name):
	#
	# This is a simple test to make sure the element treats gaps correctly
	#

	in_rate = 10		# Hz
	out_rate = 1		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	gap_frequency = 0.1	# Hz
	gap_threshold = 0.1	# Hz
	control_dump_filename = "control.dump"

	head = test_common.gapped_test_src_ints(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)
	head = tee = pipeparts.mktee(pipeline, head)
	
	head = pipeparts.mkgeneric(pipeline, tee, "lal_logical_undersampler", required_on = 0x1, status_out = 0x7)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-int, rate=%d" % int(out_rate))
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline

def lal_logical_undersampler_03(pipeline, name):
	#
	# This test reads in a gwf file that has certain bits off to test
	# the logic of the element
	# Note: To run this test you must first make a frame cache, which can be done with the
	# following command:
	# 	ls *.gwf | lalapps_path2cache > frame.cache
	#

	out_rate = 1
	
	src = pipeparts.mklalcachesrc(pipeline, location = "frame.cache", cache_dsc_regex = "L1")
	demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = True, skip_bad_files = True)
	head = pipeparts.mkqueue(pipeline, None)
	pipeparts.src_deferred_link(demux, "L1:TEST-CHANNEL", head.get_pad("sink"))
	head = tee = pipeparts.mktee(pipeline, head)
	
	head = pipeparts.mkgeneric(pipeline, tee, "lal_logical_undersampler", required_on = 0x1, status_out = 0x7)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-int, rate=%d" % int(out_rate))
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


test_common.build_and_run(lal_logical_undersampler_01, "lal_logical_undersampler_01")
#test_common.build_and_run(lal_logical_undersampler_02, "lal_logical_undersampler_02")
#test_common.build_and_run(lal_logical_undersampler_03, "lal_logical_undersampler_03")

