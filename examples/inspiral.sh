# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache"
GPSSTART="874100000000000000"
GPSSTART="874106958000000000"
GPSSTOP="874120000000000000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"
REFERENCEPSD="reference_psd.txt"
TEMPLATEBANK="H1-TMPLTBANK_09_1.207-874000000-2048.xml"

SRC="lal_framesrc \
	blocksize=$((16384*8*16)) \
	location=${LALCACHE} \
	instrument=${INSTRUMENT} \
	channel-name=${CHANNEL} \
	start-time-gps-ns=${GPSSTART} \
	stop-time-gps-ns=${GPSSTOP}"

FAKESRC="audiotestsrc \
	timestamp-offset=${GPSSTART} \
	samplesperbuffer=$((16384*16)) \
	num-buffers=$((($GPSSTOP-$GPSSTART)/100000000/16)) \
	wave=5 \
	volume=1e-20 \
! audio/x-raw-float, width=64, rate=16384, instrument=${INSTRUMENT}, channel_name=${CHANNEL}, units=strain"

INJECTIONS="lal_simulation \
	xml-location=impulse_at_874107198.xml"

WHITEN="lal_whiten \
	psd-mode=1 \
	filter-length=4 \
	convolution-length=16 \
	average-samples=64 \
	compensation-psd=${REFERENCEPSD}"

SCOPE="queue ! lal_multiscope trace-duration=4.0 frame-interval=0.0625 average-interval=32.0 do-timestamp=false ! ffmpegcolorspace ! cairotimeoverlay ! autovideosink"

FAKESINK="queue ! fakesink sync=false preroll-queue-len=1"

PLAYBACK="adder ! audioresample ! audioconvert ! audio/x-raw-float, width=32 ! audioamplify amplification=5e-2 ! audioconvert ! queue max-size-time=3000000000 ! alsasink"

#NXYDUMP="queue ! lal_nxydump start-time=106000000000 stop-time=126000000000 ! filesink sync=false preroll-queue-len=1 location"
NXYDUMP="queue ! lal_nxydump start-time=0 stop-time=384000000000 ! filesink sync=false preroll-queue-len=1 location"
#NXYDUMP="queue ! lal_nxydump start-time=235000000000 stop-time=290000000000 ! filesink sync=false preroll-queue-len=1 location"
#NXYDUMP="queue ! lal_nxydump start-time=874107188000000000 stop-time=874107208000000000 ! filesink sync=false preroll-queue-len=1 location"

#
# run with GST_DEBUG_DUMP_DOT_DIR set to some location to get a set of dot
# graph files dumped there showing the pipeline graph and the data formats
# on each link
#

gst-launch --gst-debug-level=1 \
	${SRC} \
	! progressreport \
		name=progress_src \
	! ${WHITEN} \
	! tee name=hoft_16384 \
	! audiowsinclimit \
		length=301 \
		cutoff=1024 \
	! audioresample \
	! audio/x-raw-float, rate=2048 \
	! tee name=hoft_2048 \
	hoft_16384. ! audiowsinclimit \
		length=301 \
		cutoff=256 \
	! audioresample \
	! audio/x-raw-float, rate=512 \
	! tee name=hoft_512 \
	hoft_16384. ! audiowsinclimit \
		length=301 \
		cutoff=128 \
	! audioresample \
	! audio/x-raw-float, rate=256 \
	! tee name=hoft_256 \
	hoft_16384. ! audiowsinclimit \
		length=301 \
		cutoff=64 \
	! audioresample \
	! audio/x-raw-float, rate=128 \
	! tee name=hoft_128 \
	lal_adder name=orthogonal_snr_sum_squares ! audio/x-raw-float, rate=2048 ! ${NXYDUMP}=sumsquares.txt \
	lal_adder name=snr ! audio/x-raw-float, rate=2048 ! progressreport name=progress_snr ! ${FAKESINK} \
	hoft_2048. ! queue max-size-time=50000000000 ! lal_templatebank \
		name=templatebank0 \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=0 \
		t-end=1 \
		t-total-duration=45 \
		snr-length=$((2048*1)) \
	templatebank0.sumofsquares ! audioresample filter-length=3 ! queue ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr0 \
	templatebank0.matrix ! snr0.matrix \
	templatebank0.src ! tee name=orthosnr0 ! snr0.sink \
	snr0. ! audioresample filter-length=3 ! queue ! snr. \
	hoft_512. ! queue max-size-time=50000000000 ! lal_templatebank \
		name=templatebank1 \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=1 \
		t-end=5 \
		t-total-duration=45 \
		snr-length=$((512*1)) \
	templatebank1.sumofsquares ! audioresample filter-length=3 ! queue ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr1 \
	templatebank1.matrix ! snr1.matrix \
	templatebank1.src ! tee name=orthosnr1 ! snr1.sink \
	snr1. ! audioresample filter-length=3 ! queue ! snr. \
	hoft_256. ! queue max-size-time=50000000000 ! lal_templatebank \
		name=templatebank2 \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=5 \
		t-end=13 \
		t-total-duration=45 \
		snr-length=$((256*1)) \
	templatebank2.sumofsquares ! audioresample filter-length=3 ! queue ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr2 \
	templatebank2.matrix ! snr2.matrix \
	templatebank2.src ! tee name=orthosnr2 ! snr2.sink \
	snr2. ! audioresample filter-length=3 ! queue ! snr. \
	hoft_128. ! queue max-size-time=50000000000 ! lal_templatebank \
		name=templatebank3 \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=13 \
		t-end=29 \
		t-total-duration=45 \
		snr-length=$((128*1)) \
	templatebank3.sumofsquares ! audioresample filter-length=3 ! queue ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr3 \
	templatebank3.matrix ! snr3.matrix \
	templatebank3.src ! tee name=orthosnr3 ! snr3.sink \
	snr3. ! audioresample filter-length=3 ! queue ! snr. \
	hoft_128. ! queue max-size-time=50000000000 ! lal_templatebank \
		name=templatebank4 \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=29 \
		t-end=45 \
		t-total-duration=45 \
		snr-length=$((128*1)) \
	templatebank4.sumofsquares ! audioresample filter-length=3 ! queue ! orthogonal_snr_sum_squares. \
	lal_matrixmixer \
		name=snr4 \
	templatebank4.matrix ! snr4.matrix \
	templatebank4.src ! tee name=orthosnr4 ! snr4.sink \
	snr4. ! audioresample filter-length=3 ! queue ! snr. \

exit
