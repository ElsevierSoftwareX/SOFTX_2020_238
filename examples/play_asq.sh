# A gst-launch pipeline to play AS_Q through a sound card.

LALCACHE="/home/kipp/scratch_local/874000000-20063/cache/data.cache"
GPSSTART="874000000"
GPSSTOP="874020000"
INSTRUMENT="H1"
CHANNEL="LSC-AS_Q"

gst-launch \
	lal_framesrc \
		blocksize=65536 \
		location="${LALCACHE}" \
		instrument="${INSTRUMENT}" \
		channel-name="${CHANNEL}" \
		start-time-gps="${GPSSTART}" \
		stop-time-gps="${GPSSTOP}" \
	! queue \
	! audioconvert \
	! audio/x-raw-float, width=64 \
	! audiochebband \
		lower-frequency=40 \
		upper-frequency=2500 \
		poles=8 \
	! audioconvert \
	! audio/x-raw-float, width=32 \
	! audioamplify \
		amplification=3e-1 \
	! audioconvert \
	! alsasink
