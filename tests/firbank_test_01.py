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
# is the firbank element an identity transform when given a unit impulse?
# in and out timeseries should be identical modulo start/stop transients
#


def firbank_test_01a(pipeline):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 2.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	fir_length = 21	# samples

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, gap_frequency = 200, gap_threshold = .8)
	head = tee = pipeparts.mktee(pipeline, head)

	fir_matrix = numpy.zeros((1, fir_length), dtype = "double")
	latency = (fir_length - 1) / 2
	fir_matrix[0, latency] = 1.0

	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = fir_matrix, latency = -latency)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "firbank_test_01a_out.txt")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "firbank_test_01a_in.txt")

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


test_common.build_and_run(firbank_test_01a, "firbank_test_01a")

