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
from gstlal.elements.check_timestamps import mkchecktimestamps


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


def resample_test_01a(pipeline):
	#
	# try changing these.  test should still work!
	#

	in_rate = 512	# Hz
	out_rate = 2048	# Hz
	quality = 9	# [0, 9]
	gap_frequency = None	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 20.0	# seconds

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "resample_test_01a_control.dump")
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % out_rate)
	head = mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "resample_test_01a_out.dump")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "resample_test_01a_in.dump")

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


test_common.build_and_run(resample_test_01a, "resample_test_01a")

