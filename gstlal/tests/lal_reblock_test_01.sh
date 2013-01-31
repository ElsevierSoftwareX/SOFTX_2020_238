function run_lal_reblock() {
	gst-launch audiotestsrc blocksize=16384 num-buffers=4096 wave=9 ! audio/x-raw-float, width=32, channels=1, rate=125 ! lal_reblock block-duration=0312400000 ! queue ! lal_checktimestamps silent=true timestamp-fuzz=0 ! fakesink
}

! run_lal_reblock 2>&1 | grep -q "lal_checktimestamps"
