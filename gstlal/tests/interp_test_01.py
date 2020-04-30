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
# check for timestamp drift in the audioresample
#


def interp_test_01(pipeline, name):
	#
	# try changing these.  test should still work!
	#

	in_rate = 64	# Hz
	out_rate = 128 # Hz
	gap_frequency = 0.5	# Hz
	gap_threshold = .75	# of 1
	buffer_length = 5.0	# seconds
	test_duration = 20.0	# seconds
	frequency     = 11.0
	wave          = 0

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, freq=frequency, wave=wave, buffer_length = buffer_length, rate = in_rate, width = 32, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudioconvert(pipeline, head), "audio/x-raw, format=F32LE, rate=%d" % in_rate)
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkinterpolator(pipeline, head), "audio/x-raw, format=F32LE, rate=%d" % out_rate)
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


test_common.build_and_run(interp_test_01, "interp_test_01")

