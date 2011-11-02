gst-launch \
	audiotestsrc wave=sine volume=1 freq=0.505 samplesperbuffer=262144 \
	! audio/x-raw-float,rate=16384 \
	! lal_timeseriesplotter \
	! video/x-raw-rgb,width=800,height=600 \
	! ximagesink sync=0
