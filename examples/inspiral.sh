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

SINK="queue max-size-time=96 ! lal_multiscope trace-duration=4.0 frame-interval=0.0625 average-interval=32.0 do-timestamp=false ! ffmpegcolorspace ! cairotimeoverlay ! autovideosink"
#SINK="queue ! fakesink sync=false preroll-queue-len=1"

PLAYBACK="adder ! audioconvert ! audio/x-raw-float, width=32 ! audioamplify amplification=1e-3 ! audioconvert ! queue max-size-time=3 ! progressreport update-freq=1 ! alsasink"


#
# run with GST_DEBUG_DUMP_DOT_DIR set to some location to get a set of dot
# graph files dumped there showing the pipeline graph and the data formats
# on each link
#

gst-launch --gst-debug-level=1 \
	$SRC \
	! progressreport \
	! lal_simulation \
		xml-location="bns_injections.xml" \
	! lal_whiten \
		psd-mode=1 \
		filter-length=4 \
		convolution-length=16 \
		average-samples=256 \
	! tee name=hoft_white \
	! audioresample \
	! audio/x-raw-float, rate=2048 \
	! tee name=hoft_2048 \
	! audioresample \
	! audio/x-raw-float, rate=1024 \
	! tee name=hoft_1024 \
	lal_adder name=orthogonal_snr_sum_squares ! $SINK \
	hoft_2048. ! $SINK \
	hoft_2048. ! lal_templatebank \
		name=templatebank0 \
		t-start=0 \
		t-end=1 \
		snr-length=$((2048*8)) \
	lal_matrixmixer \
		name=snr0 \
	templatebank0.src ! tee name=orthosnr0 ! $SINK \
	templatebank0.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank0.matrix ! snr0.matrix \
	orthosnr0. ! snr0.sink \
	snr0. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_2048. ! lal_templatebank \
		name=templatebank1 \
		t-start=1 \
		t-end=2 \
		snr-length=$((2048*8)) \
	lal_matrixmixer \
		name=snr1 \
	templatebank1.src ! tee name=orthosnr1 ! $SINK \
	templatebank1.sumofsquares ! queue max-size-time=64 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank1.matrix ! snr1.matrix \
	orthosnr1. ! snr1.sink \
	snr1. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_1024. ! lal_templatebank \
		name=templatebank2 \
		t-start=2 \
		t-end=4 \
		snr-length=$((1024*8)) \
	lal_matrixmixer \
		name=snr2 \
	templatebank2.src ! tee name=orthosnr2 ! $SINK \
	templatebank2.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank2.matrix ! snr2.matrix \
	orthosnr2. ! snr2.sink \
	snr2. ! queue ! fakesink sync=false preroll-queue-len=1 \

exit

	! audioresample \
	! audio/x-raw-float, rate=512 \
	! tee name=hoft_512 \
	! audioresample \
	! audio/x-raw-float, rate=256 \
	! tee name=hoft_256 \
	! audioresample \
	! audio/x-raw-float, rate=128 \
	! tee name=hoft_128 \
	hoft_512. ! lal_templatebank \
		name=templatebank3 \
		t-start=4 \
		t-end=8 \
		snr-length=$((512*8)) \
	lal_matrixmixer \
		name=snr3 \
	templatebank3.src ! tee name=orthosnr3 ! $SINK \
	templatebank3.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank3.matrix ! snr3.matrix \
	orthosnr3. ! snr3.sink \
	snr3. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_256. ! lal_templatebank \
		name=templatebank4 \
		t-start=8 \
		t-end=16 \
		snr-length=$((256*8)) \
	lal_matrixmixer \
		name=snr4 \
	templatebank4.src ! tee name=orthosnr4 ! $SINK \
	templatebank4.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank4.matrix ! snr4.matrix \
	orthosnr4. ! snr4.sink \
	snr4. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_128. ! lal_templatebank \
		name=templatebank5 \
		t-start=16 \
		t-end=32 \
		snr-length=$((128*8)) \
	lal_matrixmixer \
		name=snr5 \
	templatebank5.src ! tee name=orthosnr5 ! $SINK \
	templatebank5.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank5.matrix ! snr5.matrix \
	orthosnr5. ! snr5.sink \
	snr5. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_128. ! lal_templatebank \
		name=templatebank6 \
		t-start=32 \
		t-end=48 \
		snr-length=$((128*8)) \
	lal_matrixmixer \
		name=snr6 \
	templatebank6.src ! tee name=orthosnr6 ! $SINK \
	templatebank6.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank6.matrix ! snr6.matrix \
	orthosnr6. ! snr6.sink \
	snr6. ! queue ! fakesink sync=false preroll-queue-len=1 \
	hoft_128. ! lal_templatebank \
		name=templatebank7 \
		t-start=48 \
		t-end=64 \
		snr-length=$((128*8)) \
	lal_matrixmixer \
		name=snr7 \
	templatebank7.src ! tee name=orthosnr7 ! $SINK \
	templatebank7.sumofsquares ! queue max-size-time=96 ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank7.matrix ! snr7.matrix \
	orthosnr7. ! snr7.sink \
	snr7. ! queue ! fakesink sync=false preroll-queue-len=1 \
