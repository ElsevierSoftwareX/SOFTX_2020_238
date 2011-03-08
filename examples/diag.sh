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

function test_gate() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_out.txt" \
		audiotestsrc freq=13 samplesperbuffer=1024 num-buffers=1 \
		! audio/x-raw-float, rate=1024 \
		! tee name=control \
		! gate.control \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 \
		! tee name=orig \
		! gate.sink \
		orig. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_in.txt" \
		control. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_control.txt"
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
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! tee name=orig \
		! audioresample \
		! audio/x-raw-float, width=64, rate=16383 \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=15.8 samplesperbuffer=1025 num-buffers=8 \
		! audio/x-raw-float, width=64, rate=2047 \
		! tee name=control \
		! gate.control \
		audiotestsrc freq=256 wave=sine samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, channels=1, width=64, rate=2047 \
		! gate.sink \
		control. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_control.txt" \
		orig. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_in.txt"
}

function test_down_resampler_gaps() {
	gst-launch \
		lal_gate name=gate threshold=0.7 \
		! tee name=orig \
		! audioresample \
		! audio/x-raw-float, width=64, rate=16383 \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=15.8 samplesperbuffer=1024 num-buffers=128 \
		! audio/x-raw-float, width=64, rate=1023 \
		! tee name=control \
		! gate.control \
		audiotestsrc freq=256 wave=sine samplesperbuffer=1024 num-buffers=128 \
		! audio/x-raw-float, channels=1, width=64, rate=1023 \
		! gate.sink \
		control. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_control.txt" \
		orig. \
		! lal_nxydump \
		! queue ! filesink buffer-mode=2 location="dump_in.txt"
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
		audiotestsrc wave=9 volume=1e-21 timestamp-offset=900000000000000000 num-buffers=30 samplesperbuffer=16384 \
		! audio/x-raw-float, channels=1, width=64, rate=16384 \
		! taginject tags="instrument=\"H1\",channel-name=\"LSC-STRAIN\",units=\"strain\"" \
		! lal_simulation xml-location="test_inspiral_injections_1s_step.xml" \
		! audioamplify clipping-method=3 amplification=1e20 \
		! adder ! audioconvert ! autoaudiosink
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
                        location="/home/kipp/gwf/cache" \
                        instrument="H1" \
                        channel-name="LSC-STRAIN" \
                        num-buffers=1000 \
                        name=framesrc \
                ! lal_simulation xml-location="/home/dkeppel/lloid/HL-INJECTIONS_1_BNS_INJ-873247900-10.xml" \
                ! audioresample ! audio/x-raw-float, rate=2048 \
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
		! lalautochisq template-bank="../src/utilities/chirpmass-1.126557_H1-TMPLTBANK_02-873250008-2048-first.xml.gz" reference-psd="reference_psd.txt"\
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

test_gate
