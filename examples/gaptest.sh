gst-launch \
	lal_gate \
		name=gate \
		threshold=0.5 \
	! audioresample \
	! audio/x-raw-float, width=64, rate=128 \
	! lal_nxydump \
		start-time=0 \
		stop-time=1000000000 \
	! filesink \
		location=output.txt \
		buffer-mode=2 \
	audiotestsrc \
		wave=0 \
		freq=30.5 \
	! audio/x-raw-float, width=64, rate=512 \
	! tee \
		name=signal \
	! queue \
	! lal_nxydump \
		start-time=0 \
		stop-time=1000000000 \
	! filesink \
		location=input.txt \
		buffer-mode=2 \
	signal. \
	! queue \
	! gate.sink \
	audiotestsrc \
		wave=0 \
		freq=3 \
	! audio/x-raw-float, width=64, rate=512 \
	! tee \
		name=control \
	! queue \
	! lal_nxydump \
		start-time=0 \
		stop-time=1000000000 \
	! filesink \
		location=control.txt \
		buffer-mode=2 \
	control. \
	! queue \
	! gate.control
