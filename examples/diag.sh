function test_adder() {
	gst-launch \
		audiotestsrc freq=16 samplesperbuffer=1024 num-buffers=8 timestamp-offset=1000000000 \
		! lal_adder name=adder sync=true \
		! audio/x-raw-float, width=64, rate=16384 \
		! lal_nxydump start-time=0 stop-time=2250000000 \
		! filesink location="dump.txt" \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 timestamp-offset=1250000000 \
		! adder.
}

function test_gate() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! audio/x-raw-float, width=64, rate=16384 \
		! lal_nxydump start-time=0 stop-time=1000000000 \
		! queue ! filesink location="dump_out.txt" \
		audiotestsrc freq=13 samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, rate=1024 \
		! gate.control \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 \
		! tee name=orig \
		! gate.sink
		orig. \
		! lal_nxydump start-time=0 stop-time=1000000000 \
		! queue ! filesink location="dump_in.txt"
}

function test_resampler() {
	gst-launch \
		audiotestsrc \
		! audio/x-raw-float, width=64, rate=16384 \
		! gstlal-audioresample \
		! audio/x-raw-float, rate=4096 \
		! fakesink
}

function test_resampler_gaps() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! audioresample \
		! audio/x-raw-float, width=64, rate=2048 \
		! lal_nxydump start-time=0 stop-time=1000000000 \
		! queue ! filesink location="dump_out.txt" \
		audiotestsrc freq=13 samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, rate=1024 \
		! gate.control \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, width=64, rate=16384 \
		! tee name=orig \
		! gate.sink \
		orig. \
		! lal_nxydump start-time=0 stop-time=1000000000 \
		! queue ! filesink location="dump_in.txt"
}

function test_whiten() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-5 \
		! audio/x-raw-float, width=64, rate=1024, units="strain" \
		! lal_whiten psd-mode=1 zero-pad=0 fft-length=8 median-samples=7 average-samples=128 \
		! lal_nxydump start-time=0 stop-time=1200000000000 \
		! progressreport \
		! filesink location="dump.txt" buffer-mode=2
}

test_resampler_gaps
