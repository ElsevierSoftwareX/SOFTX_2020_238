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

seekevent = gst.event_new_seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT,
	gst.SEEK_TYPE_SET, 0, gst.SEEK_TYPE_SET, 1000 * gst.SECOND)

# Create a new source element
src = gst.element_factory_make('lal_fakeligosrc')
src.set_property('instrument', 'H1')
src.set_property('channel-name', 'LSC-STRAIN')

if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
	raise RuntimeError, "Element %s did not want to enter ready state" % src.get_name()
if not src.send_event(seekevent):
	raise RuntimeError, "Element %s did not handle seek event" % src.get_name()

# Create a new sink element
sink = gst.element_factory_make('filesink')
sink.set_property('location', '/dev/null')

# Construct pipeline
pipeline = gst.Pipeline()
pipeline.add_many(src, sink)
gst.element_link_many(src, sink)

print "Setting state to PLAYING:", pipeline.set_state(gst.STATE_PLAYING)

# Start runloop
mainloop = gobject.MainLoop()
mainloop.run()
