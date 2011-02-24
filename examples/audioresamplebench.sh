#!/bin/bash

CHANNELS=256
SAMPLESPERBUFFER=512
WIDTH=64
DATARATE=$((CHANNELS * WIDTH / 8))
BUFSIZE=$((DATARATE * SAMPLESPERBUFFER))
NUMBUFFERS=16

for UPSAMPLEPOWER in {0..5}
do
	UPSAMPLEFACTOR=$((1 << UPSAMPLEPOWER))
	FILENAME=bench.${UPSAMPLEFACTOR}.dat
	>$FILENAME
	for QUALITY in {0..10}
	do
		echo -n quality=${QUALITY}...
		echo -n "$QUALITY " >>$FILENAME
		(time -p gst-launch \
			fakesrc num-buffers=$NUMBUFFERS sizetype=fixed sizemax=$BUFSIZE filltype=zero datarate=$DATARATE \
			! audio/x-raw-float,rate=1,channels=$CHANNELS,width=$WIDTH \
			! audioresample quality=$QUALITY \
			! audio/x-raw-float,rate=$UPSAMPLEFACTOR \
			! fakesink sync=0 async=0 \
			> /dev/null) 2>&1 | sed -n "s/real \(.*\)/\1/p" >>$FILENAME
		echo done.
	done
done

python <<EndPython

import matplotlib
matplotlib.use('agg')
from pylab import *

filter_length=[8, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256]

for filtlen in filter_length:
	line = axvline(filtlen, color='k', alpha=0.25)

for upsamplepower in range(6):
	upsamplefactor = 2**upsamplepower
	filename = "bench.%d.dat" % upsamplefactor
	dat = loadtxt(filename)
	quality = dat[:,0]
	filtlen = [filter_length[int(q)] for q in quality]
	runtime = 1e6 * dat[:,1] / (upsamplefactor * $SAMPLESPERBUFFER * $NUMBUFFERS * $CHANNELS)
	plot(filtlen, runtime, label='upsampling %dX' % upsamplefactor)
legend(loc='upper right')
xlabel('filter length')
ylabel('overhead (microseconds / output sample)')
title('audioresample benchmark: overhead for different filter lengths')
savefig('bench.png')

EndPython
