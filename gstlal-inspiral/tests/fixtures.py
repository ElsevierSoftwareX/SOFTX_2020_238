"""

Test fixtures for GStreamer pipelines.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__copyright__    = "Copyright 2010, Leo Singer"



import unittest
import sys
from gstlal.pipeutil import *


class PipelineTestFixture(unittest.TestCase):
	"""Test fixture for launching a single pipeline with a GObject main loop.

	Example::

		class MyTest(PipelineTestFixture):

			def runTest(self):
				... # Build pipeline
				self.set_state(gst.STATE_PLAYING)
				self.mainloop.run()
				... # Evaluate some tests

	"""

	def setUp(self):
		self.pipeline = gst.Pipeline()
		self.mainloop = gobject.MainLoop()
		self.msg_tupl = None
		bus = self.pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

	def on_message(self, bus, message):
		if message.type == gst.MESSAGE_EOS:
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			self.msg_tupl = (gerr.domain, gerr.code, gerr.message, dbgmsg)
			print("error (%s:%d '%s'): %s" % self.msg_tupl, file=sys.stderr)
			self.mainloop.quit()

	def tearDown(self):
		try:
			if self.pipeline.set_state(gst.STATE_NULL) != gst.STATE_CHANGE_SUCCESS:
				raise RuntimeError, "Pipeline did not enter NULL state"
		finally:
			if self.msg_tupl is not None:
				raise RuntimeError, "last GStreamer error (%s:%d '%s'): %s" % self.msg_tupl
