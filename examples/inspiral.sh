# A gst-launch pipeline to perform an inspiral matched filter analysis.

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
	! audioconvert \
	! audio/x-raw-float, width=64 \
	! audiochebband \
		lower-frequency=20 \
		upper-frequency=1000 \
		poles=8 \
	! audioresample \
	! audio/x-raw-float, rate=2048 \
	! lal_templatebank \
		name=orthosnr \
		t-start=0 \
		t-end=2 \
	orthosnr.src0000 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0001 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0002 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0003 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0004 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0005 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0006 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0007 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0008 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink \
	orthosnr.src0009 ! queue ! audioconvert ! monoscope ! ffmpegcolorspace ! ximagesink
