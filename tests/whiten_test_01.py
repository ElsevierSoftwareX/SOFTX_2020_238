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
from gstlal.pipeparts import gst
import test_common


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

	def psd_resolution_changed(elem, pspec, ignored):
		delta_f = elem.get_property("delta-f")
		f_nyquist = elem.get_property("f-nyquist")
		n = int(round(f_nyquist / delta_f) + 1)
		elem.set_property("mean-psd", numpy.zeros((n,), dtype="double") + 2.0 * delta_f * (4 / f_nyquist))

	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 4.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 9)
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 1, zero_pad = zero_pad, fft_length = fft_length)
	head.connect_after("notify::f-nyquist", psd_resolution_changed, None)
	head.connect_after("notify::delta-f", psd_resolution_changed, None)
	head = pipeparts.mknofakedisconts(pipeline, head)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "whiten_test_01a_out.dump")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee, max_size_time = int(fft_length * gst.SECOND)), "whiten_test_01a_in.dump")

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
	fft_length = 4.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 10000.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 6)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 0, zero_pad = zero_pad, fft_length = fft_length)
	head = pipeparts.mknofakedisconts(pipeline, head)
	head = pipeparts.mkchecktimestamps(pipeline, head)
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

