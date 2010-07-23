#!/bin/bash

gst-launch \
	audiotestsrc freq=.05 wave=sine samplesperbuffer=128 ! audio/x-raw-float,rate=4096 ! queue ! lal_stripchart samplesperbuffer=262144 ! video/x-raw-rgb,width=600,height=200,framerate=8/1 ! videobox top=0 bottom=-400 border-alpha=0.0 ! videomixer name="mix" ! queue ! ffmpegcolorspace ! ximagesink \
	audiotestsrc freq=.05 wave=sine samplesperbuffer=128 ! audio/x-raw-float,rate=4096 ! queue ! lal_stripchart samplesperbuffer=262144 ! video/x-raw-rgb,width=600,height=200,framerate=8/1 ! videobox top=-200 bottom=-200 border-alpha=0.0 ! mix. \
	audiotestsrc freq=.05 wave=sine samplesperbuffer=128 ! audio/x-raw-float,rate=4096 ! queue ! lal_stripchart samplesperbuffer=262144 ! video/x-raw-rgb,width=600,height=200,framerate=8/1 ! videobox top=-400 bottom=0 border-alpha=0.0 ! mix.
