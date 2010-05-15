#!/usr/bin/env python
"""
Example gst-python applicaiton for trying out new ideas
"""

from gstlal.pipeutil import *

gps_start_time = 956858656

pipeline = gst.Pipeline()
src = mkelem('lal_onlinehoftsrc', {'instrument':'H1'})
pipeline.add(src)

src.send_event(gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT,
	gst.SEEK_TYPE_SET, gps_start_time * gst.SECOND,
	gst.SEEK_TYPE_NONE, -1))


snk = mkelem('fakesink')
pipeline.add(snk)
src.link(snk)

print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)

# Start runloop
mainloop = gobject.MainLoop()
mainloop.run()
