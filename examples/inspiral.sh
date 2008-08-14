# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874000000-20063/cache/data.cache"
GPSSTART="874000000"
#GPSSTOP="874020000"
GPSSTOP="874000320"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

VISUALIZE_SNR="lal_multiscope ! ffmpegcolorspace ! ximagesink"
#VISUALIZE_SNR="fakesink"

gst-launch \
	lal_framesrc \
		blocksize=$((16384*8*16)) \
		location="${LALCACHE}" \
		instrument="${INSTRUMENT}" \
		channel-name="${CHANNEL}" \
		start-time-gps="${GPSSTART}" \
		stop-time-gps="${GPSSTOP}" \
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
	templatebank0.orthogonal_snr ! $VISUALIZE_SNR
