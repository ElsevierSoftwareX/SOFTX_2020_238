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

SINK="queue max-size-time=96 ! lal_multiscope trace-duration=0.25 average-interval=10.0 ! ffmpegcolorspace ! timeoverlay ! xvimagesink"
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
	! queue ! audioresample \
	! audio/x-raw-float, rate=1024 \
	! tee name=hoft_1024 \
	! queue ! audioresample \
	! audio/x-raw-float, rate=512 \
	! tee name=hoft_512 \
	! queue ! audioresample \
	! audio/x-raw-float, rate=256 \
	! tee name=hoft_256 \
	! queue ! audioresample \
	! audio/x-raw-float, rate=128 \
	! tee name=hoft_128 \
	adder name=orthogonal_snr_sum_squares ! $SINK \
	hoft_2048. ! $SINK \
	lal_templatebank \
		name=templatebank0 \
		t-start=0 \
		t-end=1 \
		snr-length=$((2048*8)) \
	hoft_2048. ! queue ! templatebank0.sink \
	templatebank0.orthogonal_snr ! queue ! tee name=orthosnr0 ! $SINK \
	templatebank0.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_templatebank \
		name=templatebank1 \
		t-start=1 \
		t-end=2 \
		snr-length=$((2048*8)) \
	hoft_2048. ! queue ! templatebank1.sink \
	templatebank1.orthogonal_snr ! queue ! tee name=orthosnr1 ! $SINK \
	templatebank1.orthogonal_snr_sum_squares ! queue max-size-time=64 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_templatebank \
		name=templatebank2 \
		t-start=2 \
		t-end=4 \
		snr-length=$((1024*8)) \
	hoft_1024. ! queue ! templatebank2.sink \
	templatebank2.orthogonal_snr ! queue ! tee name=orthosnr2 ! $SINK \
	templatebank2.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_templatebank \
		name=templatebank3 \
		t-start=4 \
		t-end=8 \
		snr-length=$((512*8)) \
	hoft_512. ! queue ! templatebank3.sink \
	templatebank3.orthogonal_snr ! queue ! tee name=orthosnr3 ! $SINK \
	templatebank3.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_templatebank \
		name=templatebank4 \
		t-start=8 \
		t-end=16 \
		snr-length=$((256*8)) \
	hoft_256. ! queue ! templatebank4.sink \
	templatebank4.orthogonal_snr ! queue ! tee name=orthosnr4 ! $SINK \
	templatebank4.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_templatebank \
		name=templatebank5 \
		t-start=16 \
		t-end=32 \
		snr-length=$((128*8)) \
	hoft_128. ! queue ! templatebank5.sink \
	templatebank5.orthogonal_snr ! queue ! tee name=orthosnr5 ! $SINK \
	templatebank5.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \

	#lal_templatebank \
	#	name=templatebank6 \
	#	t-start=32 \
	#	t-end=48 \
	#	snr-length=$((128*8)) \
	#hoft_128. ! queue ! templatebank6.sink \
	#templatebank6.orthogonal_snr ! queue ! tee name=orthosnr6 ! $SINK \
	#templatebank6.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	#lal_templatebank \
	#	name=templatebank7 \
	#	t-start=48 \
	#	t-end=64 \
	#	snr-length=$((128*8)) \
	#hoft_128. ! queue ! templatebank7.sink \
	#templatebank7.orthogonal_snr ! queue ! tee name=orthosnr7 ! $SINK \
	#templatebank7.orthogonal_snr_sum_squares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \

	#orthosnr0. ! queue ! templatebank0.orthogonal_snr_sink \
	#orthosnr1. ! queue ! templatebank1.orthogonal_snr_sink \
	#orthosnr2. ! queue ! templatebank2.orthogonal_snr_sink \
	#orthosnr3. ! queue ! templatebank3.orthogonal_snr_sink \
	#orthosnr4. ! queue ! templatebank4.orthogonal_snr_sink \
	#orthosnr5. ! queue ! templatebank5.orthogonal_snr_sink \
	#orthosnr5. ! queue ! templatebank6.orthogonal_snr_sink \
	#orthosnr7. ! queue ! templatebank7.orthogonal_snr_sink \
	#templatebank0.snr ! queue max-size-time=96 ! fakesink \
	#templatebank1.snr ! queue max-size-time=96 ! fakesink \
	#templatebank2.snr ! queue max-size-time=96 ! fakesink \
	#templatebank3.snr ! queue max-size-time=96 ! fakesink \
	#templatebank4.snr ! queue max-size-time=96 ! fakesink \
	#templatebank5.snr ! queue max-size-time=96 ! fakesink \
	#templatebank6.snr ! queue max-size-time=96 ! fakesink \
	#templatebank7.snr ! queue max-size-time=96 ! fakesink
