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
# make pretty plots
#


def channelgram_test_01a(pipeline):
	#
	# try changing these.  test should still work!
	#

	wave = 0	# gst-inspect audiotestsrc
	freq = 25	# Hz
	rate = 2048	# Hz
	gap_frequency = None	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 20.0	# seconds
	width = 640	# pixels
	height = 480	# pixels
	framerate = "%d/%d" % (4, 1)	# frames/second

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = wave, freq = freq, gap_frequency = gap_frequency, gap_threshold = gap_threshold)
	head = pipeparts.mkprogressreport(pipeline, head, "src")
	head = pipeparts.mkchannelgram(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "video/x-raw-rgb, width=%d, height=%d, framerate=%s" % (width, height, framerate))
	pipeparts.mkogmvideosink(pipeline, pipeparts.mkqueue(pipeline, head), "channelgram_test_01a_out.avi")

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


test_common.build_and_run(channelgram_test_01a, "channelgram_test_01a")

