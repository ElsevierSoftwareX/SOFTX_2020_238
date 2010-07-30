#!/usr/bin/env python

from optparse import OptionParser, Option

opts, args = OptionParser(
	option_list=[
		Option("--gps-start-time","-s",type="int"),
	]
).parse_args()

from gstlal.pipeutil import *
from gstlal.lloidparts import mkelems_fast

pipeline = gst.Pipeline()
elems = mkelems_fast(pipeline,
	"lal_onlinehoftsrc", {"instrument": "L1"},
	"audioresample", {"gap-aware": True},
	"capsfilter", {"caps": gst.Caps("audio/x-raw-float,rate=2048,width=64")},
	"fakesink",
	"lal_stripchart", {"samplesperbuffer": 2048*8, "y-min": -1e-16, "y-max": 1e-16},
	"capsfilter", {"caps": gst.Caps("video/x-raw-rgb,width=1200,height=200,framerate=16/1")},
	"ximagesink"
)

pipeline.set_state(gst.STATE_READY)
elems[0].send_event(gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT | gst.SEEK_FLAG_FLUSH,
	gst.SEEK_TYPE_SET, opts.gps_start_time * gst.SECOND,
	gst.SEEK_TYPE_NONE, -1))

mainloop = gobject.MainLoop()
pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
