#!/bin/sh

pass_through () {
	gst-launch gds_lvshmsrc shm-name="testing" ! fdsink fd=1 &
	gst-launch fdsrc fd=0 ! gds_lvshmsink shm-name="testing"
	kill $!
}

pass_through <lvshmsinksrc_test_01.sh | cmp lvshmsinksrc_test_01.sh
