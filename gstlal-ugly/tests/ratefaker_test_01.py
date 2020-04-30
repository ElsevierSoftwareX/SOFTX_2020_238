#!/usr/bin/env python3

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


#
# test rate faker
#


def ratefaker_test_01a(pipeline, dummy):
	#
	# try changing these.  test should still work!
	#

	wave = 0	# gst-inspect audiotestsrc
	freq = 25	# Hz
	in_rate = 2048	# Hz
	out_rate = 1024	# Hz
	gap_frequency = 1.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 200.0	# seconds

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, wave = wave, freq = freq, gap_frequency = gap_frequency, gap_threshold = gap_threshold)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "in.txt")
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src")
	head = pipeparts.mkgeneric(pipeline, head, "audioratefaker")
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % out_rate)
	pipeparts.mknxydumpsink(pipeline, head, "out.txt")

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


test_common.build_and_run(ratefaker_test_01a, "ratefaker_test_01a")

