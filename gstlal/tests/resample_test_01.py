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


def resample_test_01a(pipeline, name):
	#
	# try changing these.  test should still work!
	#

	in_rate = 2048	# Hz
	out_rate = 512	# Hz
	quality = 9	# [0, 9]
	gap_frequency = None	# Hz
	gap_threshold = 0.0	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 20.0	# seconds

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	elem = pipeparts.Gst.ElementFactory.make("audiocheblimit", None)
	elem.set_property("mode", 0)
	elem.set_property("cutoff", .95 * out_rate / 2.0)
	pipeline.add(elem)
	head.link(elem)
	head = elem
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw, rate=%d" % out_rate)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline


#
# check for timestamp drift in the audioresample
#


def resample_test_01b(pipeline, name):
	#
	# try changing these.  test should still work!
	#

	in_rate = 512	# Hz
	out_rate = 2048	# Hz
	quality = 9	# [0, 9]
	gap_frequency = None	# Hz
	gap_threshold = 0.0	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 20.0	# seconds

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = in_rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "%s_control.dump" % name)
	elem = pipeparts.Gst.ElementFactory.make("audiocheblimit", None)
	elem.set_property("mode", 0)
	elem.set_property("cutoff", .95 * out_rate / 2.0)
	pipeline.add(elem)
	head.link(elem)
	head = elem
	head = tee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw, rate=%d" % out_rate)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw, rate=%d" % in_rate)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw, rate=%d" % out_rate)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw, rate=%d" % in_rate)
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


test_common.build_and_run(resample_test_01a, "resample_test_01a")
test_common.build_and_run(resample_test_01b, "resample_test_01b")

