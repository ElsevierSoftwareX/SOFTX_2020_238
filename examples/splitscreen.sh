#!/bin/bash

gst-launch \
	audiotestsrc freq=8.1 wave=triangle do-timestamp=1 ! audio/x-raw-float,rate=4096 ! lal_timeseriesplotter ! queue ! video/x-raw-rgb,width=600,height=200,framerate=4/1 ! videobox top=0 bottom=-400 border-alpha=0.0 ! videomixer name="mix" ! ffmpegcolorspace ! ximagesink \
	audiotestsrc freq=8.1 wave=triangle do-timestamp=1 ! audio/x-raw-float,rate=4096 ! lal_timeseriesplotter ! queue ! video/x-raw-rgb,width=600,height=200,framerate=4/1 ! videobox top=-200 bottom=-200 border-alpha=0.0 ! mix. \
	audiotestsrc freq=8.1 wave=triangle do-timestamp=1 ! audio/x-raw-float,rate=4096 ! lal_timeseriesplotter ! queue ! video/x-raw-rgb,width=600,height=200,framerate=4/1 ! videobox top=-400 bottom=0 border-alpha=0.0 ! mix.

#gst-launch \
#	videotestsrc ! video/x-raw-rgb,width=400,height=200 ! videobox right=-400 bottom=-200 border-alpha=0.0 ! alpha ! queue ! videomixer name="mix" ! ffmpegcolorspace ! ximagesink \
#	videotestsrc ! video/x-raw-rgb,width=400,height=200 ! videobox left=-400 bottom=-200 border-alpha=0.0 ! alpha ! queue ! mix. \
#	videotestsrc ! video/x-raw-rgb,width=400,height=200 ! videobox right=-400 top=-200 border-alpha=0.0 ! alpha ! queue ! mix. \
#	videotestsrc ! video/x-raw-rgb,width=400,height=200 ! videobox left=-400 top=-200 border-alpha=0.0 ! alpha ! queue ! mix.
