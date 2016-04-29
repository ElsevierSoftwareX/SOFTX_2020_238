#!/bin/sh

smname="gstlal-test"

pass_through () {
	gst-launch-1.0 --quiet fdsrc fd=0 ! gds_lvshmsink buffer-mode=1 shm-name=$smname gds_lvshmsrc shm-name=$smname num-buffers=1 ! fdsink fd=1 sync=false async=false
	smkill $smname 2>/dev/null
}

pass_through <${srcdir:-.}/lvshmsinksrc_test_01.sh | cmp ${srcdir:-.}/lvshmsinksrc_test_01.sh
