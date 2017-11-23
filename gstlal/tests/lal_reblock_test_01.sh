run_lal_reblock () {
	NAME=$1
	shift
	FORMAT=${1:-F32LE}
	shift
	CHANNELS=${1:-1}
	shift
	echo "$NAME: format=$FORMAT, channels=$CHANNELS"
	gst-launch-1.0 audiotestsrc blocksize=16384 num-buffers=512 wave=9 ! audio/x-raw, format=${FORMAT}, channels=${CHANNELS}, rate=300 ! tee name=in ! lal_reblock block-duration=0312400000 ! queue ! lal_checktimestamps ! lal_nxydump ! filesink location="${NAME}_out.dump" async=false in. ! lal_nxydump ! filesink location="${NAME}_in.dump" async=false
}

name="lal_reblock_test_01"
echo === Running Test ${name} ===
{ ! run_lal_reblock ${name}a F32LE 1 2>&1 | grep -q "lal_checktimestamps" ; } && ${srcdir:-.}/cmp_nxydumps.py --compare-exact-gaps ${name}a_{in,out}.dump
{ ! run_lal_reblock ${name}b F64LE 1 2>&1 | grep -q "lal_checktimestamps" ; } && ${srcdir:-.}/cmp_nxydumps.py --compare-exact-gaps ${name}b_{in,out}.dump
{ ! run_lal_reblock ${name}c F32LE 2 2>&1 | grep -q "lal_checktimestamps" ; } && ${srcdir:-.}/cmp_nxydumps.py --compare-exact-gaps ${name}c_{in,out}.dump
{ ! run_lal_reblock ${name}d F64LE 2 2>&1 | grep -q "lal_checktimestamps" ; } && ${srcdir:-.}/cmp_nxydumps.py --compare-exact-gaps ${name}d_{in,out}.dump
