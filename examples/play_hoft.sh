# A gst-launch pipeline to play h(t) through a sound card.

LALCACHE="/home/kipp/scratch_local/874000000-20063/cache/data.cache"
GPSSTART="874000000"
GPSSTOP="874020000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

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
		lower-frequency=45 \
		upper-frequency=2500 \
		poles=8 \
	! audioconvert \
	! audio/x-raw-float, width=32 \
	! audioamplify \
		amplification=2e+17 \
	! audioconvert \
	! alsasink
