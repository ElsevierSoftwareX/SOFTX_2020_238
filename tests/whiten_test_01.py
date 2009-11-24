#!/usr/bin/python


import numpy
import sys


import gobject
import pygst
pygst.require("0.10")
import gst


from gstlal import pipeparts


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
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 5, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=64, rate=%d" % rate)
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 1, zero_pad = zero_pad, fft_length = fft_length)
	gobject.add_emission_hook(head, "delta-f-changed", delta_f_changed, None)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "whiten_test_01a_out.txt")
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee), "whiten_test_01a_in.txt")


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


class Handler(object):
	def __init__(self, mainloop, pipeline):
		self.mainloop = mainloop
		self.pipeline = pipeline
		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

	def on_message(self, bus, message):
		if message.type == gst.MESSAGE_EOS:
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			print >>sys.stderr, "error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()


gobject.threads_init()

mainloop = gobject.MainLoop()

pipeline = gst.Pipeline("whiten_test_01a")

whiten_test_01a(pipeline)

handler = Handler(mainloop, pipeline)

pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
