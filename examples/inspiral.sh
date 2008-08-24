# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache"
GPSSTART="874100000"
GPSSTART="874106958"
GPSSTOP="874120000"
#GPSSTOP="874103600"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

SRC="lal_framesrc \
	blocksize=$((16384*8*16)) \
	location=${LALCACHE} \
	instrument=${INSTRUMENT} \
	channel-name=${CHANNEL} \
	start-time-gps=${GPSSTART} \
	stop-time-gps=${GPSSTOP}"

#SRC="fakesrc \
#	blocksize=$((16384*8*16)) \
#	num-buffers=$((320/16)) \
#	sizetype=2 \
#	sizemax=$((16384*8*16)) \
#	filltype=2 \
#	datarate=$((16384*8)) \
#! audio/x-raw-float, width=64, channels=1, rate=16384, endianness=1234"

SINK="queue ! lal_multiscope trace-duration=0.25 ! ffmpegcolorspace ! timeoverlay ! ffmpegcolorspace ! ximagesink"
#SINK="queue ! fakesink"

gst-launch --gst-debug-level=1 \
	$SRC \
	! audioconvert \
	! audio/x-raw-float, width=64 \
	! audiochebband \
		lower-frequency=20 \
		upper-frequency=1000 \
		poles=8 \
	! audioresample \
	! audio/x-raw-float, rate=2048 \
	! lal_templatebank \
		name=templatebank0 \
		t-start=0 \
		t-end=1 \
		snr-length=$((2048*16)) \
	templatebank0.orthogonal_snr ! $SINK \
	templatebank0.bank_magnitude ! $SINK
