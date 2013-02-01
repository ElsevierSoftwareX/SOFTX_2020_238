name=lal_reblock_test_01

function run_lal_reblock() {
	gst-launch audiotestsrc blocksize=16384 num-buffers=512 wave=9 ! tee name=in ! audio/x-raw-float, width=32, channels=1, rate=300 ! lal_reblock block-duration=0312400000 ! queue ! lal_checktimestamps silent=true timestamp-fuzz=1 ! lal_nxydump ! filesink location="${name}_out.dump" sync=false async=false in. ! lal_nxydump ! filesink location="${name}_in.dump" sync=false async=false
}

echo === Running Test ${name} ===
{ ! run_lal_reblock 2>&1 | grep -q "lal_checktimestamps" ; } &&
./cmp_nxydumps ${name}_{in,out}.dump
