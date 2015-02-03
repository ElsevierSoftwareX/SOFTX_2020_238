#!/usr/bin/env python

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst


from gstlal import pipeparts

pipeline = gst.Pipeline("test_postcoh")
mainloop = gobject.MainLoop()

src1 = pipeparts.mkaudiotestsrc(pipeline, wave = 9)
src1 = pipeparts.mkcapsfilter(pipeline, src1, "audio/x-raw-float, width=32, channels=2, rate=4096")
src2 = pipeparts.mkaudiotestsrc(pipeline, wave = 9)
src2 = pipeparts.mkcapsfilter(pipeline, src2, "audio/x-raw-float, width=32, channels=2, rate=4096")
src3 = pipeparts.mkaudiotestsrc(pipeline, wave = 9)
src3 = pipeparts.mkcapsfilter(pipeline, src3, "audio/x-raw-float, width=32, channels=2, rate=4096")


postcoh = gst.element_factory_make("cuda_postcoh")
postcoh.set_property("detrsp-fname", "L1H1V1_skymap.xml")
postcoh.set_property("autocorrelation-fname", "L1:H1bank.xml.gz,H1:H1bank.xml.gz,V1:H1bank.xml.gz")
postcoh.set_property("hist-trials", 1)
postcoh.set_property("snglsnr-thresh", 1.0)
pipeline.add(postcoh)
src1.link_pads(None, postcoh, "H1")
src2.link_pads(None, postcoh, "L1")
src3.link_pads(None, postcoh, "V1")

sink = gst.element_factory_make("postcoh_filesink")
sink.set_property("location", "postcoh_table.xml.gz")
sink.set_property("compression", 1)
pipeline.add(sink)
postcoh.link(sink)

pipeline.set_state(gst.STATE_PLAYING)

mainloop.run()

