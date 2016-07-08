#!/bin/sh

# check for bad discont flags in output of audioresample by generating a
# stream with 1 sample per buffer, and then downsampling to a lower sample
# rate so that some input buffers produce no output buffers.  the correct
# behaviour is for it to produce buffers with no data, but old versions
# produced no buffer followed by a buffer marked as a discont (due to
# unavoidable behaviour in the GstBaseTransform class from which it is
# derived).

run_test() {
	gst-launch-1.0 \
		audiotestsrc wave=0 samplesperbuffer=1 num-buffers=1024 \
		! audio/x-raw,rate=512,format=F64LE,channels=1 \
		! audioresample \
		! audio/x-raw,rate=128 \
		! lal_checktimestamps \
		! fakesink
}

! run_test 2>&1 | grep -q "lal_checktimestamps"
