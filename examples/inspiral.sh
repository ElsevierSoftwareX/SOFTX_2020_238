# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache"
GPSSTART="874100000000000000"
GPSSTART="874106958000000000"
GPSSTOP="874120000000000000"
#GPSSTOP="874110558000000000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

SRC="lal_framesrc \
	blocksize=$((16384*8*16)) \
	location=${LALCACHE} \
	instrument=${INSTRUMENT} \
	channel-name=${CHANNEL} \
	start-time-gps-ns=${GPSSTART} \
	stop-time-gps-ns=${GPSSTOP}"

FAKESRC="audiotestsrc \
	timestamp-offset=$GPSSTART \
	samplesperbuffer=$((16384*16)) \
	num-buffers=$((($GPSSTOP-$GPSSTART)/100000000)) \
	wave=5 \
	volume=1e-20 \
! audio/x-raw-float, width=64, rate=16384, instrument=${INSTRUMENT}, channel=${CHANNEL}, units=count"

WHITEN="lal_whiten \
	psd-mode=1 \
	filter-length=4 \
	convolution-length=16 \
	average-samples=256 \
	compensation-psd=reference_psd.txt"

SINK="queue ! lal_multiscope trace-duration=4.0 frame-interval=0.0625 average-interval=32.0 do-timestamp=false ! ffmpegcolorspace ! cairotimeoverlay ! autovideosink"

FAKESINK="queue ! fakesink sync=false preroll-queue-len=1"
SINK=$FAKESINK

PLAYBACK="adder ! audioresample ! audioconvert ! audio/x-raw-float, width=32 ! audioamplify amplification=1e-3 ! audioconvert ! queue max-size-time=3000000000 ! alsasink"

#NXYDUMP="queue ! lal_nxydump start-time=110000000000 stop-time=130000000000 ! filesink sync=false preroll-queue-len=1 location"
NXYDUMP="queue ! lal_nxydump start-time=170000000000 stop-time=310000000000 ! filesink sync=false preroll-queue-len=1 location"

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
	! $WHITEN \
	! audioresample \
		filter-length=64 \
	! audio/x-raw-float, rate=2048 \
	! tee name=hoft_2048 \
	! audioresample \
		filter-length=64 \
	! audio/x-raw-float, rate=1024 \
	! tee name=hoft_1024 \
	! audioresample \
		filter-length=64 \
	! audio/x-raw-float, rate=512 \
	! tee name=hoft_512 \
	! audioresample \
		filter-length=64 \
	! audio/x-raw-float, rate=256 \
	! tee name=hoft_256 \
	! audioresample \
		filter-length=64 \
	! audio/x-raw-float, rate=128 \
	! tee name=hoft_128 \
	lal_adder name=orthogonal_snr_sum_squares ! $NXYDUMP=nxydump.txt \
	hoft_2048. ! $SINK \
	hoft_2048. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank0 \
		t-start=0 \
		t-end=1 \
		snr-length=$((2048*8)) \
	templatebank0.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank0.src ! tee name=orthosnr0 ! $SINK \
	lal_matrixmixer \
		name=snr0 \
	templatebank0.matrix ! snr0.matrix \
	orthosnr0. ! snr0.sink \
	snr0. ! $FAKESINK \
	hoft_2048. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank1 \
		t-start=1 \
		t-end=2 \
		snr-length=$((2048*8)) \
	templatebank1.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank1.src ! tee name=orthosnr1 ! $SINK \
	lal_matrixmixer \
		name=snr1 \
	templatebank1.matrix ! snr1.matrix \
	orthosnr1. ! snr1.sink \
	snr1. ! $FAKESINK \
	hoft_1024. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank2 \
		t-start=2 \
		t-end=4 \
		snr-length=$((1024*8)) \
	templatebank2.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank2.src ! tee name=orthosnr2 ! $SINK \
	lal_matrixmixer \
		name=snr2 \
	templatebank2.matrix ! snr2.matrix \
	orthosnr2. ! snr2.sink \
	snr2. ! $FAKESINK \
	hoft_512. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank3 \
		t-start=4 \
		t-end=8 \
		snr-length=$((512*8)) \
	templatebank3.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank3.src ! tee name=orthosnr3 ! $SINK \
	lal_matrixmixer \
		name=snr3 \
	templatebank3.matrix ! snr3.matrix \
	orthosnr3. ! snr3.sink \
	snr3. ! $FAKESINK \
	hoft_256. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank4 \
		t-start=8 \
		t-end=16 \
		snr-length=$((256*8)) \
	templatebank4.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank4.src ! tee name=orthosnr4 ! $SINK \
	lal_matrixmixer \
		name=snr4 \
	templatebank4.matrix ! snr4.matrix \
	orthosnr4. ! snr4.sink \
	snr4. ! $FAKESINK \
	hoft_128. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank5 \
		t-start=16 \
		t-end=32 \
		snr-length=$((128*8)) \
	templatebank5.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	templatebank5.src ! tee name=orthosnr5 ! $SINK \
	lal_matrixmixer \
		name=snr5 \
	templatebank5.matrix ! snr5.matrix \
	orthosnr5. ! snr5.sink \
	snr5. ! $FAKESINK \
	hoft_128. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank6 \
		t-start=32 \
		t-end=48 \
		snr-length=$((128*8)) \
	templatebank6.src ! tee name=orthosnr6 ! $SINK \
	templatebank6.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr6 \
	templatebank6.matrix ! snr6.matrix \
	orthosnr6. ! snr6.sink \
	snr6. ! $FAKESINK \
	hoft_128. ! queue max-size-time=96000000000 ! lal_templatebank \
		name=templatebank7 \
		t-start=48 \
		t-end=64 \
		snr-length=$((128*8)) \
	templatebank7.src ! tee name=orthosnr7 ! $SINK \
	templatebank7.sumofsquares ! queue ! audioresample ! audio/x-raw-float, rate=2048 ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr7 \
	templatebank7.matrix ! snr7.matrix \
	orthosnr7. ! snr7.sink \
	snr7. ! $FAKESINK \
