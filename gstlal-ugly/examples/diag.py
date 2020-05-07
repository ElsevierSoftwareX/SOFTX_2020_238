#!/usr/bin/env python3


import sys
import numpy

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from ligo import segments
from pylal.datatypes import LIGOTimeGPS
from gstlal import pipeparts


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


def test_histogram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, None, "src")
	pipeparts.src_deferred_link(pipeparts.mkframecppchanneldemux(pipleine, pipeparts.mkcachesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", cache_src_regex = "H.*")), "H1:LSC-STRAIN", head.get_pad("sink"))
	head = pipeparts.mkwhiten(pipeline, head)
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkspectrumplot(pipeline, head.get_pad("mean-psd")), "video/x-raw-rgb, width=640, height=480")))
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/4")))


def test_channelgram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, None, "src")
	pipeparts.src_deferred_link(pipeparts.mkframecppchanneldemux(pipleine, pipeparts.mkcachesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", cache_src_regex = "H.*")), "H1:LSC-STRAIN", head.get_pad("sink"))
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
	segs = [(40, 50), (20, 40), (2, 3)]
	head = pipeparts.mksegmentsrc(pipeline, segments.segmentlist(segments.segment((LIGOTimeGPS(a), LIGOTimeGPS(b))) for (a, b) in segs))
	head = pipeparts.mkprogressreport(pipeline, head, "progress")
	head = pipeparts.mkaudioconvert(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "test.txt")


def test_timeslicechisq(pipeline):
	timeslicesnrs = []

	chifacs = [0.250570, 0.307837, 0.315783, 0.566514, 0.5599827729939556, 0.31425027633241631, 0.095521268663057449]

	for fac in chifacs:
		timeslicesnr = gst.element_factory_make("audiotestsrc")
		timeslicesnr.set_property("samplesperbuffer", 1024)
		timeslicesnr.set_property("num-buffers", 100)
		timeslicesnr.set_property("wave", 9)
		timeslicesnr.set_property("volume", 1.)
		pipeline.add(timeslicesnr)
		timeslicesnr = pipeparts.mkcapsfilter(pipeline, timeslicesnr, "audio/x-raw-float, width=64, rate=2048, channels=2")
		# turn 2 (independent) real channels into 2 (identical) complex channels
		mixmatrix = [[fac, 0, fac, 0],
			     [0, fac, 0, fac]]
		timeslicesnr = pipeparts.mkmatrixmixer(pipeline, timeslicesnr, mixmatrix)

		timeslicesnr = pipeparts.mktogglecomplex(pipeline, timeslicesnr)
		timeslicesnr = pipeparts.mktee(pipeline, timeslicesnr)
		timeslicesnrs.append(timeslicesnr)

	timeslicechisq = gst.element_factory_make("lal_timeslicechisq")
	pipeline.add(timeslicechisq)
	for timeslicesnr in timeslicesnrs:
		pipeparts.mkqueue(pipeline, timeslicesnr).link(timeslicechisq)
	# we have 2 complex channels so we need 2 chifacs per time slice
	timeslicechisq.set_property("chifacs-matrix", [[fac**2., fac**2.] for fac in chifacs])
	timeslicechisq = pipeparts.mkqueue(pipeline, timeslicechisq)
	timeslicechisq = pipeparts.mkprogressreport(pipeline, timeslicechisq, 'timeslicechisq')

	for n,timeslicesnr in enumerate(timeslicesnrs):
		timeslicesnr = pipeparts.mktogglecomplex(pipeline, timeslicesnr)		
		pipeparts.mknxydumpsink(pipeline, timeslicesnr, "dump_timeslicesnr%i.txt"%(n))

	# output will be 2 real channels, each chisq distributed with dof=2*len(chifacs)-2
	pipeparts.mknxydumpsink(pipeline, timeslicechisq, "dump_timeslicechisq.txt")

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
			print("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg), file=sys.stderr)
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()


gobject.threads_init()

mainloop = gobject.MainLoop()

pipeline = gst.Pipeline("diag")

test_segmentsrc(pipeline)

handler = Handler(mainloop, pipeline)

if pipeline.set_state(gst.STATE_PAUSED) == gst.STATE_CHANGE_FAILURE:
	raise RuntimeError("pipeline failed to enter PAUSED state")
pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, LIGOTimeGPS(874000000).ns(), gst.SEEK_TYPE_SET, LIGOTimeGPS(874020000).ns())
if pipeline.set_state(gst.STATE_PLAYING) == gst.STATE_CHANGE_FAILURE:
	raise RuntimeError("pipeline failed to enter PLAYING state")
mainloop.run()
