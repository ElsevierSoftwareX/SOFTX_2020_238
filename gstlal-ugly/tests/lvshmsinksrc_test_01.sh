#!/bin/sh

smname="gstlal-test"

pass_through () {
	gst-launch --quiet fdsrc fd=0 ! gds_lvshmsink buffer-mode=1 shm-name=$smname gds_lvshmsrc shm-name=$smname num-buffers=1 ! fdsink fd=1 sync=false async=false
	smkill $smname 2>/dev/null
}

pass_through <lvshmsinksrc_test_01.sh | cmp lvshmsinksrc_test_01.sh
