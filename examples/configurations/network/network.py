#!/usr/bin/env python
__author__ = "Leo Singer <leo.singer@ligo.org>"


from optparse import Option, OptionParser
opts, args = OptionParser(
	option_list =
	[
		Option("--instrument", "-i", metavar="IFO", action="append", help="Instruments to analyze."),
		Option("--gps-start-time", "-s", metavar="INT", type="int", help="GPS time at which to start analysis."),
		Option("--gps-end-time", "-e", metavar="INT", type="int", help="GPS time at which to end analysis."),
	]
).parse_args()


opts.psd_fft_length = 8


from gstlal.gstlal_svd_bank import read_bank
from gstlal.gstlal_reference_psd import read_psd
from gstlal import lloidparts
from gstlal.pipeutil import gst, gobject
from gstlal.lloidparts import mkelems_fast
import sys


# FIXME: This class, or something like it, occurs in many of our pipelines.
# Maybe we could put a generalized version of it in a library?
class LLOIDHandler(object):
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
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
			sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))


pipeline = gst.Pipeline()
mainloop = gobject.MainLoop()
handler = LLOIDHandler(mainloop, pipeline)


seekevent = gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT,
	gst.SEEK_TYPE_SET, opts.gps_start_time * gst.SECOND,
	gst.SEEK_TYPE_SET, opts.gps_end_time * gst.SECOND
)


coincstage = mkelems_fast(
	pipeline,
	"lal_coinc",
	"fakesink"
)


for ifo in opts.instrument:
	bank = read_bank("bank.%s.pickle" % ifo)
	bank.logname = ifo # FIXME This is only need to give elements names, that should be automatic.
	psd = read_psd("reference_psd.%s.xml.gz" % ifo)
	rates = bank.get_rates()

	basicsrc = lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, ifo, None, online_data=True)
	hoftdict = lloidparts.mkLLOIDsrc(pipeline, basicsrc, rates, psd=psd, psd_fft_length=opts.psd_fft_length)
	branch = lloidparts.mkLLOIDsingle(pipeline, hoftdict, ifo, bank, lloidparts.mkcontrolsnksrc(pipeline, max(rates)))
	branch.link_pads("src", coincstage[0], "sink%d")


pipeline.set_state(gst.STATE_PLAYING)
gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "network")
mainloop.run()
