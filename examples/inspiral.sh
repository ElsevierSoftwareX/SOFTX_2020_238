# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/channa/scratch/frames/S5/strain-L2/LLO/L-L1_RDS_C03_L2-8741/L.cache"
GPSSTART="874100000000000000"
GPSSTART="874106958000000000"
GPSSTOP="874120000000000000"
INSTRUMENT="L1"
CHANNEL="LSC-STRAIN"
REFERENCEPSD="reference_psd.txt"
TEMPLATEBANK="H1-TMPLTBANK_09_1.207-874000000-2048.xml"
SUMSQUARESTHRESHOLD="2.0"
SNRTHRESHOLD="1.0"
TRIGGERFILENAME="output.xml"

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

FAKESINK="queue ! fakesink sync=false preroll-queue-len=1"

SCOPE="queue ! lal_multiscope trace-duration=4.0 frame-interval=0.0625 average-interval=32.0 do-timestamp=false ! ffmpegcolorspace ! cairotimeoverlay ! autovideosink"

PLAYBACK="adder ! gstlal-audioresample ! audioconvert ! audio/x-raw-float, width=32 ! audioamplify amplification=5e-2 ! audioconvert ! queue max-size-time=3000000000 ! alsasink"

INJECTIONS="lal_simulation \
	xml-location=bns_injections.xml"

WHITEN="lal_whiten \
	psd-mode=1 \
	zero-pad=4 \
	fft-length=16 \
	average-samples=64 \
	compensation-psd=${REFERENCEPSD}"

function templatebank() {
	SUFFIX=${1}
	TSTART=${2}
	TEND=${3}
	TTOTALDURATION=${4}
	SNRLENGTH=${5}
	echo "lal_templatebank \
		name=templatebank${SUFFIX} \
		template-bank=${TEMPLATEBANK} \
		reference-psd=${REFERENCEPSD} \
		t-start=${TSTART} \
		t-end=${TEND} \
		t-total-duration=${TTOTALDURATION} \
		snr-length=${SNRLENGTH} \
	templatebank${SUFFIX}.sumofsquares ! gstlal-audioresample ! queue ! orthogonal_snr_sum_squares_adder. \
	lal_gate \
		name=snr_gate${SUFFIX} \
		threshold=${SUMSQUARESTHRESHOLD} \
	orthogonal_snr_sum_squares. ! queue ! snr_gate${SUFFIX}.control \
	templatebank${SUFFIX}.src ! tee name=orthogonalsnr${SUFFIX} ! queue ! snr_gate${SUFFIX}.sink \
	lal_matrixmixer \
		name=mixer${SUFFIX} \
	! tee \
		name=snr${SUFFIX} \
	! gstlal-audioresample quality=0 ! queue ! snradder. \
	templatebank${SUFFIX}.matrix ! tee name=matrix${SUFFIX} ! queue ! mixer${SUFFIX}.matrix \
	snr_gate${SUFFIX}.src ! mixer${SUFFIX}.sink \
	lal_chisquare \
		name=chisquare${SUFFIX} \
	! gstlal-audioresample quality=0 ! queue ! chisquareadder. \
	matrix${SUFFIX}. ! queue ! chisquare${SUFFIX}.matrix \
	templatebank${SUFFIX}.chifacs ! queue ! chisquare${SUFFIX}.chifacs \
	orthogonalsnr${SUFFIX}. ! queue ! chisquare${SUFFIX}.orthosnr \
	snr${SUFFIX}. ! queue ! chisquare${SUFFIX}.snr"
}

# output for hardware injection at 874107078.149271066
#NXYDUMP="queue ! lal_nxydump start-time=874107068000000000 stop-time=874107088000000000 ! filesink sync=false preroll-queue-len=1 buffer-mode=2 location"
# alternate output for use with injection (bns_injections.xml=874107198.405080859 and impulse_at_874107198.xml) at 874107189
NXYDUMP="queue ! lal_nxydump start-time=874107188000000000 stop-time=874107258000000000 ! filesink sync=false preroll-queue-len=1 buffer-mode=2 location"
# alternate output to dump lots and lots of data (the whole cache)
#NXYDUMP="queue ! lal_nxydump start-time=874100128000000000 stop-time=874120000000000000 ! filesink sync=false preroll-queue-len=1 buffer-mode=2 location"
# ??
#NXYDUMP="queue ! lal_nxydump start-time=235000000000 stop-time=290000000000 ! filesink sync=false preroll-queue-len=1 buffer-mode=2 location"

#
# run with GST_DEBUG_DUMP_DOT_DIR set to some location to get a set of dot
# graph files dumped there showing the pipeline graph and the data formats
# on each link
#

gst-launch --gst-debug-level=2 \
	${SRC} \
	! progressreport \
		name=progress_src \
	! ${INJECTIONS} \
	! gstlal-audioresample ! audio/x-raw-float, rate=4096 \
	! ${WHITEN} \
	! tee name=hoft_4096 \
	hoft_4096. ! gstlal-audioresample ! audio/x-raw-float, rate=2048 ! tee name=hoft_2048 \
	hoft_4096. ! gstlal-audioresample ! audio/x-raw-float, rate=512 ! tee name=hoft_512 \
	hoft_4096. ! gstlal-audioresample ! audio/x-raw-float, rate=256 ! tee name=hoft_256 \
	hoft_4096. ! gstlal-audioresample ! audio/x-raw-float, rate=128 ! tee name=hoft_128 \
	lal_adder \
		name=orthogonal_snr_sum_squares_adder \
		sync=true \
	! audio/x-raw-float, rate=4096 \
	! tee name=orthogonal_snr_sum_squares \
	! progressreport name=progress_sumsquares \
	! ${NXYDUMP}=sumsquares.txt \
	lal_adder \
		name=snradder \
		sync=true \
	! audio/x-raw-float, rate=4096 \
	! tee name=snr \
	! progressreport name=progress_snr \
	! ${NXYDUMP}=snr.txt \
	lal_adder \
		name=chisquareadder \
		sync=true \
	! audio/x-raw-float, rate=4096 \
	! tee name=chisquare \
	! progressreport name=progress_chisquare \
	! ${NXYDUMP}=chisquare.txt \
	hoft_4096. ! queue max-size-time=50000000000 ! $(templatebank 0 0.0 0.25 45.25 $((4096*1))) \
	hoft_2048. ! queue max-size-time=50000000000 ! $(templatebank 1 0.25 1.25 45.25 $((2048*1))) \
	hoft_512. ! queue max-size-time=50000000000 ! $(templatebank 2 1.25 5.25 45.25 $((512*1))) \
	hoft_256. ! queue max-size-time=50000000000 ! $(templatebank 3 5.25 13.25 45.25 $((256*1))) \
	hoft_128. ! queue max-size-time=50000000000 ! $(templatebank 4 13.25 29.25 45.25 $((128*1))) \
	hoft_128. ! queue max-size-time=50000000000 ! $(templatebank 5 29.25 45.25 45.25 $((128*1))) \
	lal_triggergen \
		name=triggergen \
		bank-filename=${TEMPLATEBANK} \
		snr-thresh=${SNRTHRESHOLD} \
	! lal_triggerxmlwriter \
		location=${TRIGGERFILENAME} \
		sync=false \
		preroll-queue-len=1 \
	snr. ! queue ! triggergen.snr \
	chisquare. ! queue ! triggergen.chisquare \

exit

