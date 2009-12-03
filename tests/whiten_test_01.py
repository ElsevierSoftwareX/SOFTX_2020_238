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
# is the whiten element an identity transform when given a unit PSD?  in
# and out timeseries should be identical modulo FFT precision and start-up
# and shut-down transients.
#


def whiten_test_01a(pipeline):
	#
	# signal handler to construct a new unit PSD (with LAL's
	# normalization) whenever the frequency resolution or Nyquist
	# frequency changes
	#

	def delta_f_changed(elem, delta_f, ignored):
		n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
		elem.set_property("psd", numpy.zeros((n,), dtype="double") + 2.0 * delta_f)

	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 2.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 50.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration)
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 1, zero_pad = zero_pad, fft_length = fft_length)
	head.connect_after("delta-f-changed", delta_f_changed, None)
	head = mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "whiten_test_01a_out.dump")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "whiten_test_01a_in.dump")

	#
	# done
	#

	return pipeline


#
# does the whitener turn coloured Gaussian noise into zero-mean,
# unit-variance stationary white Gaussian noise?
#


def whiten_test_01b(pipeline):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 2.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration)
	head = mkchecktimestamps(pipeline, pipeparts.mkwhiten(pipeline, head, psd_mode = 0, zero_pad = zero_pad, fft_length = fft_length))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "whiten_test_01b_out.dump")

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


test_common.build_and_run(whiten_test_01a, "whiten_test_01a")
test_common.build_and_run(whiten_test_01b, "whiten_test_01b")

