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
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink location="dump_out.txt" \
		audiotestsrc freq=13 samplesperbuffer=1024 num-buffers=8 \
		! audio/x-raw-float, rate=1024 \
		! gate.control \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=8 \
		! tee name=orig \
		! gate.sink
		orig. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
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
		! tee name=orig \
		! audioresample \
		! audio/x-raw-float, width=64, rate=2048 \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink buffer-mode=2 location="dump_out.txt" \
		audiotestsrc freq=13 samplesperbuffer=1024 num-buffers=80 \
		! audio/x-raw-float, width=64, rate=1024 \
		! tee name=control \
		! gate.control \
		audiotestsrc freq=256 samplesperbuffer=1024 num-buffers=2000 \
		! audio/x-raw-float, width=64, rate=16384 \
		! gate.sink \
		control. \
		! lal_nxydump start-time=0 stop-time=1000000000 \
		! queue ! filesink buffer-mode=2 location="dump_control.txt" \
		orig. \
		! lal_nxydump start-time=0 stop-time=10000000000 \
		! queue ! filesink buffer-mode=2 location="dump_in.txt"
}

function test_whiten() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, width=64, rate=1024, units="strain" \
		! lal_whiten psd-mode=1 zero-pad=0 fft-length=8 median-samples=7 average-samples=128 \
		! lal_nxydump start-time=0 stop-time=1200000000000 \
		! progressreport \
		! filesink location="dump.txt" buffer-mode=2
}

function test_simulation() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-21 timestamp-offset=873247800000000000 \
		! audio/x-raw-float, width=64, rate=1024, units="strain" \
		! lal_simulation xml-location="/home/dkeppel/lloid/HL-INJECTIONS_1_BNS_INJ-873247900-10.xml" \
		! audioamplify clipping-method=3 amplification=1e20 \
		! lal_nxydump start-time=873247900000000000 stop-time=873247910000000000 \
		! progressreport \
		! filesink location="dump.txt" buffer-mode=2
}

function test_simulation2wav() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-21 timestamp-offset=873247900000000000 num-buffers=20 \
		! audio/x-raw-float, width=64, rate=1024, units="strain" \
		! lal_simulation xml-location="/home/dkeppel/lloid/HL-INJECTIONS_1_BNS_INJ-873247900-10.xml" \
		! progressreport \
		! audioamplify clipping-method=3 amplification=1e20 \
		! wavenc \
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

function test_pipeline() {
        gst-launch \
                audiotestsrc wave=9 volume=1e-21 timestamp-offset=873247900000000000 num-buffers=10 \
                ! audio/x-raw-float, width=64, rate=16384 \
                ! audioresample ! audio/x-raw-float, rate=2048 \
                ! progressreport \
                ! fakesink
}

function test_fakeLIGO(){
        ./LIGO_noise_test \
                --frame-cache "/home/kipp/gwf/cache" \
		--gps-start-time 800000000 \
		--gps-stop-time 800000200 \
                --instrument "H1" \
                --output "fakeLIGOpsds.xml";
        echo "</LIGO_LW>" >> fakeLIGOpsds.xml;
        python ligo_lw_test_01.py
}

function test_autochisq() {
	gst-launch \
		audiotestsrc wave=9 volume=1e-2 \
		! audio/x-raw-float, width=64, rate=2048 \
		! lalautochisq template-bank="../src/utilities/chirpmass-1.126557_H1-TMPLTBANK_02-873250008-2048-first.xml.gz" reference-psd="reference_psd.txt"\
		! progressreport \
		! fakesink
}

test_autochisq

