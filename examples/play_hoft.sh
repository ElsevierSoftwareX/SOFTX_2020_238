# A gst-launch pipeline to play h(t) through a sound card.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874000000-20000.cache"
GPSSTART="874000000"
GPSSTOP="874020000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

gst-launch --gst-debug-level=1 \
	lal_framesrc \
		blocksize=65536 \
		location="${LALCACHE}" \
		instrument="${INSTRUMENT}" \
		channel-name="${CHANNEL}" \
		start-time-gps="${GPSSTART}" \
		stop-time-gps="${GPSSTOP}" \
	! audioconvert \
	! audio/x-raw-float, width=64 \
	! audiochebband \
		lower-frequency=45 \
		upper-frequency=2500 \
		poles=8 \
	! audioconvert \
	! audio/x-raw-float, width=32 \
	! audioamplify \
		amplification=2e+17 \
	! tee name=tee \
	! queue \
	! audioconvert \
	! alsasink  \
	tee. ! queue ! audioconvert ! lal_multiscope trace-duration=0.25 average-interval=10.0 ! ffmpegcolorspace ! timeoverlay ! xvimagesink
