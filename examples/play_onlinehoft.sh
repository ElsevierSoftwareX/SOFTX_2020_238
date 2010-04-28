# A gst-launch pipeline to encode h(t) from NDS2 into a OGG/Vorbis file
# The 'adder' element is included as a hack to throw away GPS time stamps; otherwise
# the autoaudiosink won't play

gst-launch --seek=`lalapps_tconvert now`000000000 \
	lal_onlinehoftsrc \
        instrument=H1 \
	! queue min-threshold-buffers=2 \
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
