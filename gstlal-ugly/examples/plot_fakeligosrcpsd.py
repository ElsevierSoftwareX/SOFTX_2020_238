#!/usr/bin/env python3
"""
Example gst-python applicaiton for trying out new ideas
"""

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
from gstlal.pipeutil import *
from gstlal import pipeio
import sys

gps_start_time = 956858656

class PSDHandler(object):
	def __init__(self, mainloop, pipeline, verbose = False):
		self.mainloop = mainloop
		self.pipeline = pipeline
		self.verbose = verbose

		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

		self.psd = None

	def on_message(self, bus, message):
		if message.type == gst.MESSAGE_EOS:
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			print("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg), file=sys.stderr)
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MessageType.ELEMENT:
			if message.structure.get_name() == "spectrum":
				self.psd = pipeio.parse_spectrum_message(message)

# Construct pipeline
pipeline = gst.Pipeline()

# Create a new source element
elems = mkelems_in_bin(pipeline,
	(sys.argv[1],),
	('lal_whiten', {'psd-mode': 0, 'zero-pad': 0, 'fft-length': 8, 'median-samples': 7, 'average-samples': 128}),
	('fakesink',)
)

# Play pipeline
print("Setting state to PAUSED:", pipeline.set_state(gst.STATE_PAUSED))
print(pipeline.get_state())

# Seek the source
print("Seeking:", pipeline.seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH,
	gst.SEEK_TYPE_SET, 0,
	gst.SEEK_TYPE_SET, 128 * gst.SECOND))

# Start runloop
mainloop = gobject.MainLoop()

handler = PSDHandler(mainloop, pipeline, verbose=True)

# Play pipeline
print("Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING))

mainloop.run()

from pylab import *
print(psd)
loglog(arange(len(handler.psd.data)) * handler.psd.deltaF, sqrt(handler.psd.data))
xlabel('Frequency [Hz]')
ylabel('Amplitude spectral density [1/sqrt(Hz)]')
title(sys.argv[1])
show()
