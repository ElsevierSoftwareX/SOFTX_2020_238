#!/usr/bin/env python


import sys


from gstlal.pipeutil import *


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
	head = pipeparts.mkaudiochebband(pipeline, head, 45, 2500)
	pipeparts.mkplaybacksink(pipeline, head, amplification=3e16)


def test_histogram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-STRAIN"), "src")
	head = pipeparts.mkwhiten(pipeline, head)
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkspectrumplot(pipeline, head, pad = "psd"), "video/x-raw-rgb, width=640, height=480")))
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/4")))


def test_channelgram(pipeline):
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkframesrc(pipeline, location = "/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache", instrument = "H1", channel_name = "LSC-STRAIN"), "src")
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head), "audio/x-raw-float, rate=1024")

	head = pipeparts.mkwhiten(pipeline, head)

	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkqueue(pipeline, head, max_size_buffers = 5)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=4/1")
	pipeparts.mkvideosink(pipeline, pipeparts.mkqueue(pipeline, head, max_size_buffers = 5))

	pipeparts.mkplaybacksink(pipeline, pipeparts.mkaudiochebband(pipeline, tee, 40, 500), amplification = 3e16)


def test_sumsquares(pipeline):
	head = gst.element_factory_make("audiotestsrc")
	head.set_property("samplesperbuffer", 2048)
	head.set_property("wave", 9)
	head.set_property("volume", 1)
	pipeline.add(head)
	head = pipeparts.mkprogressreport(pipeline, pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=2048, channels=2"), "src")
	head = pipeparts.mksumsquares(pipeline, head)
	pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, head), "video/x-raw-rgb, width=640, height=480, framerate=1/8"))


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
			print >>sys.stderr, "error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()


gobject.threads_init()

mainloop = gobject.MainLoop()

pipeline = gst.Pipeline("diag")

test_histogram(pipeline)

handler = Handler(mainloop, pipeline)

pipeline.set_state(gst.STATE_PAUSED)
pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, LIGOTimeGPS(874000000).ns(), gst.SEEK_TYPE_SET, LIGOTimeGPS(874020000).ns())
pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
