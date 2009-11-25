#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import sys


import gobject
import pygst
pygst.require("0.10")
import gst


from gstlal import pipeparts


gobject.threads_init()


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def gapped_test_src(pipeline, buffer_length = 1.0, rate = 2048, test_duration = 10.0, gap_frequency = 1.3, gap_threshold = .8):
	src = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 5, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=64, rate=%d" % rate)
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=64, rate=%d" % rate)
	return pipeparts.mkgate(pipeline, src, threshold = gap_threshold, control = control)


#
# =============================================================================
#
#                               Pipeline Handler
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


def build_and_run(pipelinefunc, name):
	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline(name)
	handler = Handler(mainloop, pipelinefunc(pipeline))
	pipeline.set_state(gst.STATE_PLAYING)
	mainloop.run()
