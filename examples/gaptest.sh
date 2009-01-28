STOP_TIME=40000000000
IN_SAMPLE=64
OUT_SAMPLE=2048
QUALITY=10
gst-launch-0.10 \
	lal_gate \
		name=gate \
		threshold=0.75 \
	! gstlal-audioresample \
	! audio/x-raw-float, width=64, rate=${OUT_SAMPLE}, quality=${QUALITY} \
	! lal_nxydump \
		start-time=0 \
		stop-time=${STOP_TIME} \
	! filesink \
		location=output.txt \
		buffer-mode=2 \
	audiotestsrc \
		wave=0 \
		freq=20.0 \
	! audio/x-raw-float, width=64, rate=${IN_SAMPLE} \
	! tee \
		name=signal \
	! queue \
	! lal_nxydump \
		start-time=0 \
		stop-time=${STOP_TIME} \
	! filesink \
		location=input.txt \
		buffer-mode=2 \
	signal. \
	! queue \
	! gate.sink \
	audiotestsrc \
		wave=0 \
		freq=0.125 \
	! audio/x-raw-float, width=64, rate=${IN_SAMPLE} \
	! tee \
		name=control \
	! queue \
	! lal_nxydump \
		start-time=0 \
		stop-time=${STOP_TIME} \
	! filesink \
		location=control.txt \
		buffer-mode=2 \
	control. \
	! queue \
	! gate.control
