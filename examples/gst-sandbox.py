#!/usr/bin/env python
"""
Example gst-python applicaiton for trying out new ideas
"""

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require ("2.0")
import gobject
gobject.threads_init ()
import pygst
pygst.require ("0.10")
import gst

# Shouldn't need pygtk or pygst
del pygtk
del pygst

gps_start_time = 956858656

# Create a new source element
src = gst.element_factory_make('lal_onlinehoftsrc')
src.set_property('instrument', 'H1')

# Create a new sink element
sink = gst.element_factory_make('fakesink')

# Seek the source
print "Seeking:", src.seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH,
	gst.SEEK_TYPE_SET, gps_start_time * gst.SECOND,
	gst.SEEK_TYPE_SET, (gps_start_time + 16 * 5) * gst.SECOND)

# Construct pipeline
pipeline = gst.Pipeline()
pipeline.add_many(src, sink)
gst.element_link_many(src, sink)

# Play pipeline
print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)

# Start runloop
mainloop = gobject.MainLoop()
mainloop.run()
