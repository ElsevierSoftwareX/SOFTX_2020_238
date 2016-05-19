name=lal_reblock_test_01

run_lal_reblock () {
	#gst-launch-1.0 audiotestsrc blocksize=16384 num-buffers=512 wave=9 ! audio/x-raw, format=F32LE, channels=1, rate=300 ! tee name=in ! lal_reblock block-duration=0312400000 ! queue ! lal_checktimestamps silent=true timestamp-fuzz=1 ! lal_nxydump ! filesink location="${name}_out.dump" sync=false async=false in. ! lal_nxydump ! filesink location="${name}_in.dump" sync=false async=false
	gst-launch-1.0 audiotestsrc blocksize=16384 num-buffers=512 wave=9 ! audio/x-raw, format=F32LE, channels=1, rate=300 ! tee name=in ! lal_reblock block-duration=0312400000 ! queue ! lal_nxydump ! filesink location="${name}_out.dump" sync=false async=false in. ! lal_nxydump ! filesink location="${name}_in.dump" sync=false async=false
}

echo === Running Test ${name} ===
{ ! run_lal_reblock 2>&1 | grep -q "lal_checktimestamps" ; } && ${srcdir:-.}/cmp_nxydumps.py ${name}_{in,out}.dump
