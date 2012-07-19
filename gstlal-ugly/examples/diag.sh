function test_adder() {
	gst-launch \
		audiotestsrc freq=16 samplesperbuffer=1024 num-buffers=8 timestamp-offset=1000000000 \
		! lal_adder name=adder sync=true \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! lal_nxydump start-time=0 stop-time=2250000000 \
		! filesink location="dump.txt" \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 timestamp-offset=1250000000 \
		! adder.
}

function test_gate_1() {
	gst-launch \
		lal_gate name=gate threshold=0.5 attack-length=-10 hold-length=-10 invert-control=false \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_out.txt" \
		audiotestsrc volume=1 wave=3 freq=13 samplesperbuffer=1024 num-buffers=1 \
		! audio/x-raw-float, rate=1024 \
		! tee name=control \
		! gate.control \
		audiotestsrc volume=1 wave=0 freq=256 samplesperbuffer=1024 num-buffers=8 \
		! tee name=orig \
		! gate.sink \
		orig. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_in.txt" \
		control. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_control.txt"
}

function test_gate_2() {
	gst-launch \
		lal_gate name=gate threshold=0.7 leaky=false attack-length=100 hold-length=100 \
		! lal_checktimestamps ! progressreport do-query=false ! fakesink sync=false async=false \
		audiotestsrc wave=9 samplesperbuffer=16 \
		! audio/x-raw-float, channels=1, width=64, rate=6 \
		! audioresample \
		! audio/x-raw-float, rate=16000 \
		! gate.control \
		audiotestsrc wave=9 samplesperbuffer=1024 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! gate.sink
}

function test_gate_3() {
	gst-launch \
		lal_gate name=gate threshold=3 attack-length=-10 hold-length=-10 invert-control=true \
		! audio/x-raw-float, rate=1000 \
		! lal_nxydump ! filesink location="dump_out.txt" sync=false async=false \
		audiotestsrc volume=1 wave=9 samplesperbuffer=1024 num-buffers=5 \
		! tee name=tee \
		tee. ! queue ! gate.sink \
		tee. ! queue ! gate.control \
		tee. ! lal_nxydump ! filesink location="dump_in.txt" sync=false async=false
}

function test_resampler() {
	gst-launch \
		audiotestsrc wave=0 freq=1024 samplesperbuffer=10 num-buffers=1000000 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! tee name=input \
		! audioresample \
		! audio/x-raw-float, rate=4096 \
		! queue ! lal_nxydump ! filesink buffer-mode=2 location="dump_in.txt" \
		input. \
		! queue ! lal_nxydump ! filesink buffer-mode=2 location="dump_out.txt"
}

function test_up_resampler_gaps() {
	# the input to the resampler consists of intervals of a fixed
	# frequency sine function that are marked as not gaps, interleaved
	# with intervals of uniform white noise that are marked as gaps.
	# in addition to testing the resampler's handling of gaps, this
	# also tests that its output is insensitive to the data contained
	# in the input gap buffers.  the amplitude of the sine wave
	# intervals is set to a tiny fraction of the amplitude of the noise
	# intervals to help spot corruption of the output due to the noise.
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! tee name=orig \
		! lal_nxydump ! filesink sync=false async=false buffer-mode=2 location="dump_in.txt" \
		orig. \
		! audioresample \
		! audio/x-raw-float, width=64, rate=16383 \
		! lal_checktimestamps \
		! lal_nxydump ! filesink sync=false async=false buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=15.8 samplesperbuffer=1025 num-buffers=8 \
		! audio/x-raw-float, width=64, rate=2047 \
		! tee name=control \
		! lal_nxydump ! filesink sync=false async=false buffer-mode=2 location="dump_control.txt" \
		control. ! gate.control \
		lal_adder name=adder sync=true \
		! gate.sink \
		lal_gate name=srcgate1 threshold=0.7 \
		! adder. \
		lal_gate name=srcgate2 threshold=0.7 invert-control=true \
		! adder. \
		control. ! srcgate1.control \
		control. ! srcgate2.control \
		audiotestsrc freq=256 wave=sine samplesperbuffer=1024 num-buffers=8 volume=1e-6 \
		! audio/x-raw-float, channels=1, width=64, rate=2047 \
		! srcgate1.sink \
		audiotestsrc wave=white-noise samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, channels=1, width=64, rate=2047 \
		! srcgate2.sink
}

function test_down_resampler_gaps() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! tee name=orig \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_in.txt" \
		orig. \
		! audioresample \
		! audio/x-raw-float, rate=1023 \
		! lal_checktimestamps \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=4.8 samplesperbuffer=1024 num-buffers=16 \
		! audio/x-raw-float, width=64, rate=1023 \
		! tee name=control \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_control.txt" \
		control. ! gate.control \
		audiotestsrc freq=256 wave=sine samplesperbuffer=8 num-buffers=32768 \
		! audio/x-raw-float, channels=1, width=64, rate=16383 \
		! gate.sink
}

function test_whiten() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, channels=1, width=64, rate=2048 \
		! lal_whiten psd-mode=0 zero-pad=0 fft-length=8 median-samples=7 average-samples=128 \
		! lal_nxydump start-time=0 stop-time=1200000000000 \
		! progressreport \
		! filesink buffer-mode=2 location="dump.txt"
}

function test_simulation() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-21 timestamp-offset=869622009000000000 samplesperbuffer=16384 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! taginject tags="instrument=\"H1\",channel-name=\"LSC-STRAIN\",units=\"strain\"" \
		! lal_simulation xml-location="injections.xml" \
		! audioamplify clipping-method=3 amplification=1e20 \
		! adder ! audioconvert ! fakesink
}

function test_simulation2wav() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-21 timestamp-offset=900000000000000000 num-buffers=30 samplesperbuffer=16384 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! taginject tags="title=\"Inspiral Injections\",instrument=\"H1\",channel-name=\"LSC-STRAIN\",units=\"strain\"" \
		! lal_simulation xml-location="test_inspiral_injections_1s_step.xml" \
		! progressreport \
		! audioamplify clipping-method=3 amplification=1e20 \
		! wavenc \
		! id3mux write-v2=true \
		! filesink location="test_simulation2wav.wav" buffer-mode=2
}

function test_framesrc() {
        gst-launch \
                lal_framesrc \
                        blocksize=$((16384*8*16)) \
                        location="/archive/home/kipp/pbh/psd/frame.H2.cache" \
                        instrument="H2" \
                        channel-name="LSC-STRAIN" \
                        num-buffers=1000 \
                        name=framesrc \
                ! progressreport \
                ! fakesink
}

function test_fakeLIGO(){
	python plot_fakeligosrcpsd.py lal_fakeligosrc
}

function test_fakeAdvLIGO(){
	python plot_fakeligosrcpsd.py lal_fakeadvligosrc
}

function test_autochisq() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, width=64, rate=2048 \
		! lal_autochisq template-bank="../src/utilities/chirpmass-1.126557_H1-TMPLTBANK_02-873250008-2048-first.xml.gz" reference-psd="reference_psd.txt"\
		! progressreport \
		! fakesink
}

function test_nto1() {
	gst-launch \
		input-selector name=nto1 select-all=true \
		! fakesink sync=false \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, channels=1, width=64, rate=2048 \
		! queue ! nto1. \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, channels=1, width=64, rate=2048 \
		! queue ! nto1.
}

function test_chisquare_gaps() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! tee name=orig \
		! lal_autochisq template-bank=test_bank.xml reference-psd=reference_psd.txt \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=.80 samplesperbuffer=128 num-buffers=32 \
		! audio/x-raw-float, width=64, rate=2048 \
		! tee name=control \
		! gate.control \
		audiotestsrc wave=9 samplesperbuffer=128 num-buffers=32 \
		! audio/x-raw-float, channels=2, width=64, rate=2048 \
		! gate.sink \
		control. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_control.txt" \
		orig. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_in.txt"
}

function test_chained_resamplers() {
	gst-launch \
		audiotestsrc wave=9 samplesperbuffer=128 num-buffers=32 \
		! audio/x-raw-float, channels=1, width=64, rate=128 \
		! tee name=src \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_in.txt" \
		src. ! audioresample quality=4 \
		! audio/x-raw-float, rate=256 \
		! audioresample quality=4 \
		! audio/x-raw-float, rate=512 \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_out_a.txt" \
		src. ! audioresample quality=4 \
		! audio/x-raw-float, rate=512 \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_out_b.txt"

}

function test_audioundersample() {
	gst-launch \
		audiotestsrc wave=0 freq=32 samplesperbuffer=9 num-buffers=32 \
		! audio/x-raw-float, channels=1, width=64, rate=512 \
		! tee name=src \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_in.txt" \
		src. ! lal_audioundersample \
		! audio/x-raw-float, rate=128 \
		! lal_nxydump ! queue ! filesink buffer-mode=2 location="dump_out.txt"
}

function test_togglecomplex() {
	gst-launch \
		audiotestsrc wave=0 freq=32 samplesperbuffer=913 num-buffers=2048 \
		! audio/x-raw-float, channels=2, width=64, rate=512 \
		! lal_togglecomplex \
		! fakesink
}

function test_triggergen() {
	gst-launch \
		audiotestsrc wave=9 samplesperbuffer=128 \
		! audio/x-raw-float, channels=2, width=64, rate=2048 \
		! taginject tags="instrument=\"H1\",channel-name=\"LSC-STRAIN\"" \
		! lal_togglecomplex \
		! lal_triggergen name=triggergen snr-thresh=3 bank-filename="banks/femtobank.xml" \
		! progressreport \
		! fakesink \
		audiotestsrc wave=9 samplesperbuffer=128 \
		! audio/x-raw-float, channels=1, width=64, rate=2048 \
		! triggergen.
}

test_down_resampler_gaps
