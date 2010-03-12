# A gst-launch pipeline to encode h(t) from NDS2 into a OGG/Vorbis file
# The 'adder' element is included as a hack to throw away GPS time stamps; otherwise
# the autoaudiosink won't play

HOST="ldas-pcdev1.ligo.caltech.edu"
REQUESTED_CHANNEL_NAME="H1:DMT-STRAIN"
SEEK_SECONDS=952215400

gst-launch \
	ndssrc --seek=${SEEK_SECONDS}000000000 \
        host="${HOST}" \
        channel-name="${REQUESTED_CHANNEL_NAME}" \
    ! queue min-threshold-time=8000000000 \
	! audiochebband \
		lower-frequency=45 \
		upper-frequency=2500 \
		poles=8 \
	! audioamplify \
		clipping-method=3 \
		amplification=2e+17 \
	! adder \
	! audiorate \
	! audioconvert \
	! autoaudiosink
