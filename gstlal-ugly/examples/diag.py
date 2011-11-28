#!/usr/bin/env python


import sys
import numpy

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from gstlal import pipeutil

from pylal.datatypes import LIGOTimeGPS
from gstlal import pipeparts


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


def play_asq(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-AS_Q"), "src")
	head = pipeparts.mkaudiochebband(pipeline, head, 40, 2500)
	pipeparts.mkplaybacksink(pipeline, head, amplification=3e-2)


def play_hoft(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-STRAIN"), "src")
	#head = pipeparts.mkaudiochebband(pipeline, head, 50, 4096)
	#pipeparts.mkplaybacksink(pipeline, head, amplification=3e17)
	head = pipeparts.mkwhiten(pipeline, head, zero_pad = 0, fft_length = 4)
	head = pipeparts.mknofakedisconts(pipeline, head)
	pipeparts.mkplaybacksink(pipeline, head)


def test_histogram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-STRAIN"), "src")
	head = pipeparts.mkwhiten(pipeline, head)
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkspectrumplot(pipeline, head, pad = "mean-psd"), "video/x-raw-rgb, width=640, height=480")))
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/4")))


def test_channelgram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-STRAIN"), "src")
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head), "audio/x-raw-float, rate=1024")

	head = pipeparts.mkwhiten(pipeline, head)

	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkqueue(pipeline, head, max_size_buffers = 5)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=4/1")
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, head, max_size_buffers = 5))

	pipeparts.mkplaybacksink(pipeline, pipeparts.mkaudiochebband(pipeline, tee, 40, 500), amplification = 3e18)


def test_sumsquares(pipeline):
	head = gst.element_factory_make("audiotestsrc")
	head.set_property("samplesperbuffer", 2048)
	head.set_property("wave", 9)
	head.set_property("volume", 1)
	pipeline.add(head)
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=2048, channels=2"), "src")
	head = pipeparts.mksumsquares(pipeline, head)
	pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/8"))


def test_firbank(pipeline):
	head = gst.element_factory_make("audiotestsrc")
	head.set_property("samplesperbuffer", 32)
	head.set_property("wave", 0)	# sin(t)
	head.set_property("freq", 300.0)
	pipeline.add(head)

	head = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=64, rate=2048"))
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "dump_in.txt")

	head = pipeparts.mkfirbank(pipeline, pipeparts.mkqueue(pipeline, head), latency = -2, fir_matrix = [[0.0, 0.0, 1.0, 0.0, 0.0]])
	pipeparts.mknxydumpsink(pipeline, head, "dump_out.txt")


def test_segmentsrc(pipeline):
	elems = []
	segs = numpy.array([[40,50],[20,40],[2, 3]], dtype=numpy.int64) * gst.SECOND
	elems.append(pipeutil.mkelem("lal_segmentsrc", {"invert-output":False, "segment-list":segs}))
	elems.append(pipeutil.mkelem("progressreport"))
	elems.append(pipeutil.mkelem("audioconvert"))
	elems.append(pipeutil.mkelem("lal_nxydump"))
	elems.append(pipeutil.mkelem("filesink", {"location":"test.txt"}))
	for elem in elems: pipeline.add(elem)
	gst.element_link_many(*elems)


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

play_hoft(pipeline)

handler = Handler(mainloop, pipeline)

pipeline.set_state(gst.STATE_PAUSED)
pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, LIGOTimeGPS(874000000).ns(), gst.SEEK_TYPE_SET, LIGOTimeGPS(874020000).ns())
pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
