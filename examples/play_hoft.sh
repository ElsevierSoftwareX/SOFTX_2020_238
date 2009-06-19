# A gst-launch pipeline to play h(t) through a sound card.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache"
GPSSTART="874000000000000000"
GPSSTOP="874020000000000000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

# the adder element preceding the alsasink element is a hack to strip the
# time stamps from the buffers.  the framesrc element doesn't send the
# proper "new segment" event down the pipeline so the alsasink element
# doesn't know to expect a data stream starting at some huge time stamp,
# instead it waits until it sees a buffer with t=0 (it waits for the clock
# to wrap-around, it shouldn't do that but I think an integer overflow
# makes it think that 874000000000000000 precedes 0).  one day the framesrc
# element will get fixed.

gst-launch --gst-debug-level=1 \
	lal_framesrc \
		blocksize=65536 \
		location="${LALCACHE}" \
		instrument="${INSTRUMENT}" \
		channel-name="${CHANNEL}" \
		start-time-gps-ns="${GPSSTART}" \
		stop-time-gps-ns="${GPSSTOP}" \
	! audiochebband \
		lower-frequency=45 \
		upper-frequency=2500 \
		poles=8 \
	! audioamplify \
		amplification=2e+17 \
	! tee name=tee \
	! queue \
	! audioconvert \
	! adder \
	! alsasink \
	tee. ! queue ! audioconvert ! lal_multiscope trace-duration=1.0 frame-interval=0.03125 average-interval=10.0 ! ffmpegcolorspace ! timeoverlay ! xvimagesink sync=false
