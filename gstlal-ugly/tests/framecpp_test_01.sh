#!/bin/sh

CMP_NXYDUMPS=`/usr/bin/env python3 -c "import gstlal.cmp_nxydumps ; print(gstlal.cmp_nxydumps.__file__.replace('.pyc', '.py'))"`

gst-launch-1.0 audiotestsrc wave=9 samplesperbuffer=89 num-buffers=3000 ! audio/x-raw, format=F32LE, rate=1024 ! tee name="src" ! lal_nxydump ! filesink sync=false async=false location="framecpp_test_01_in.dump" src. ! .H2:FAKE-DATA framecpp_channelmux ! framecpp_channeldemux .H2:FAKE-DATA ! lal_nxydump ! filesink sync=false async=false location="framecpp_test_01_out.dump"

/usr/bin/env python3 $CMP_NXYDUMPS framecpp_test_01_in.dump framecpp_test_01_out.dump
