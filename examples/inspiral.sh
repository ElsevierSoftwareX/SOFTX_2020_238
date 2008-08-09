# A gst-launch pipeline to perform an inspiral matched filter analysis.

LALCACHE="/home/kipp/scratch_local/874000000-20063/cache/data.cache"
GPSSTART="874000000"
GPSSTOP="874020000"
INSTRUMENT="H1"
CHANNEL="LSC-STRAIN"

VISUALIZE_SNR="audioconvert ! audio/x-raw-float, width=32 ! audioamplify amplification=1e13 ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink"
#VISUALIZE_SNR="fakesink"

gst-launch \
	lal_framesrc \
		blocksize=65536 \
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
		t-end=2 \
	templatebank0.orthosnr0000 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0001 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0002 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0003 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0004 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0005 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0006 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0007 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0008 ! queue ! $VISUALIZE_SNR \
	templatebank0.orthosnr0009 ! queue ! $VISUALIZE_SNR
