# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache"
GPSSTART="874100000"
GPSSTART="874106958"
GPSSTOP="874120000"
#GPSSTOP="874110558"
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
#! audio/x-raw-float, width=64, channels=1, rate=16384, endianness=1234, instrument=${INSTRUMENT}, channel=${CHANNEL}"

SINK="queue ! lal_multiscope trace-duration=0.25 average-interval=10.0 ! ffmpegcolorspace ! timeoverlay ! xvimagesink"
#SINK="queue ! fakesink"

gst-launch --gst-debug-level=1 \
	$SRC \
	! lal_simulation \
		xml-location="bns_injections.xml" \
	! audiochebband \
		lower-frequency=40 \
		upper-frequency=1000 \
		poles=8 \
	! audioresample \
	! audio/x-raw-float, rate=2048 \
	! tee name=hoft_2048 \
	! audioresample \
	! audio/x-raw-float, rate=1024 \
	! tee name=hoft_1024 \
	! audioresample \
	! audio/x-raw-float, rate=512 \
	! tee name=hoft_512 \
	adder name=orthogonal_snr_sum_squares ! $SINK \
	hoft_2048. ! queue max-size-time=32 ! $SINK \
	lal_templatebank \
		name=templatebank0 \
		t-start=0 \
		t-end=1 \
		snr-length=$((2048*8)) \
	hoft_2048. ! queue ! templatebank0.sink \
	templatebank0.orthogonal_snr ! tee name=orthosnr0 ! $SINK \
	templatebank0.orthogonal_snr_sum_squares ! queue max-size-time=16 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	orthosnr0. ! queue max-size-time=16 ! templatebank0.orthogonal_snr_sink \
	templatebank0.snr ! queue max-size-time=16 ! fakesink \
	lal_templatebank \
		name=templatebank1 \
		t-start=1 \
		t-end=2 \
		snr-length=$((2048*8)) \
	hoft_2048. ! queue ! templatebank1.sink \
	templatebank1.orthogonal_snr ! tee name=orthosnr1 ! $SINK \
	templatebank1.orthogonal_snr_sum_squares ! queue max-size-time=16 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	orthosnr1. ! queue max-size-time=16 ! templatebank1.orthogonal_snr_sink \
	templatebank1.snr ! queue max-size-time=16 ! fakesink \
	lal_templatebank \
		name=templatebank2 \
		t-start=2 \
		t-end=4 \
		snr-length=$((1024*8)) \
	hoft_1024. ! queue ! templatebank2.sink \
	templatebank2.orthogonal_snr ! tee name=orthosnr2 ! $SINK \
	templatebank2.orthogonal_snr_sum_squares ! queue max-size-time=16 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	orthosnr2. ! queue max-size-time=16 ! templatebank2.orthogonal_snr_sink \
	templatebank2.snr ! queue max-size-time=16 ! fakesink \
	lal_templatebank \
		name=templatebank3 \
		t-start=4 \
		t-end=8 \
		snr-length=$((512*8)) \
	hoft_512. ! queue ! templatebank3.sink \
	templatebank3.orthogonal_snr ! tee name=orthosnr3 ! $SINK \
	templatebank3.orthogonal_snr_sum_squares ! queue max-size-time=16 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	orthosnr3. ! queue max-size-time=16 ! templatebank3.orthogonal_snr_sink \
	templatebank3.snr ! queue max-size-time=16 ! fakesink
