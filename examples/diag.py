#!/usr/bin/python


import sys


import gobject
import pygst
pygst.require("0.10")
import gst


from pylal.datatypes import LIGOTimeGPS
from gstlal.elements import histogram
from gstlal import pipeparts


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


def mkhistogram(pipeline, src):
	elem = histogram.Histogram()
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkcolorspace(pipeline, src):
	elem = gst.element_factory_make("ffmpegcolorspace")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkvideosink(pipeline, src):
	elem = gst.element_factory_make("autovideosink")
	pipeline.add(elem)
	src.link(elem)
	return elem


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

pipeline = gst.Pipeline("diag")
head = pipeparts.mkprogressreport(pipeline, pipeparts.mkfakesrc(pipeline, location = None, instrument = "H1", channel_name = "LSC-STRAIN"), "src")
head = pipeparts.mkcapsfilter(pipeline, mkhistogram(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/4")
mkvideosink(pipeline, mkcolorspace(pipeline, head))

handler = Handler(mainloop, pipeline)

pipeline.set_state(gst.STATE_PAUSED)
pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, 873247860000000000, gst.SEEK_TYPE_SET, 873424754000000000)
pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
