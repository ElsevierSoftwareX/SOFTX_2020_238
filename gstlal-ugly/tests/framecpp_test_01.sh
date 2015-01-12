#!/bin/sh

gst-launch audiotestsrc wave=9 samplesperbuffer=89 num-buffers=3000 ! audio/x-raw-float, width=32, rate=1024 ! tee name="src" ! lal_nxydump ! filesink sync=false async=false location="framecpp_test_01_in.dump" src. ! .H2:FAKE-DATA framecpp_channelmux ! framecpp_channeldemux .H2:FAKE-DATA ! lal_nxydump ! filesink sync=false async=false location="framecpp_test_01_out.dump"

../../gstlal/tests/cmp_nxydumps.py framecpp_test_01_in.dump framecpp_test_01_out.dump
