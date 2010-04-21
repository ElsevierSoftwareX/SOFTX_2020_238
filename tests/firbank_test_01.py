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
# is the firbank element an identity transform when given a unit impulse?
# in and out timeseries should be identical modulo start/stop transients
#


def firbank_test_01a(pipeline):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	gap_frequency = 13.0	# Hz
	gap_threshold = 0.8	# of 1
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds
	fir_length = 21	# samples
	latency = (fir_length - 1) / 2	# samples, in [0, fir_length)

	#
	# build pipeline
	#

	head = test_common.gapped_test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = "firbank_test_01a_control.dump")
	head = tee = pipeparts.mktee(pipeline, head)

	fir_matrix = numpy.zeros((1, fir_length), dtype = "double")
	fir_matrix[0, (fir_matrix.shape[1] - 1) - latency] = 1.0

	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = fir_matrix, latency = latency, time_domain = False)
	head = mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "firbank_test_01a_out.dump")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "firbank_test_01a_in.dump")

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

