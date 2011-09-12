#!/usr/bin/env python
from gstlal.pipeutil import *
from gstlal.lloidparts import mkelems_fast, LLOIDHandler
import numpy

upsample_factor = 4
num_zeros = 8

def lanczos_kernel(upsample_factor, num_zeros):
	"""Create a Lanczos interpolation filter.  upsample_factor is the ratio of
	the output rate to the input rate.  num_zeros is the number of nodes of the
	sinc function to retain."""
	i = numpy.arange(- upsample_factor * num_zeros + 1,
		upsample_factor * num_zeros, dtype=float)
	kernel = numpy.sinc(i / upsample_factor)
	kernel *= numpy.sinc(i / (upsample_factor * num_zeros))
	return kernel

kernel = lanczos_kernel(upsample_factor, num_zeros)
numpy.savetxt('kernel.txt', kernel)

inrate = 64
outrate = inrate * upsample_factor

pipeline = gst.Pipeline("impulse_response")
mainloop = gobject.MainLoop()
handler = LLOIDHandler(mainloop, pipeline)

pipeline = gst.parse_launch("""
	audiotestsrc wave=square freq=1 volume=1 samplesperbuffer=128 num-buffers=10
		! audio/x-raw-float,rate=%d,width=64,channels=1
		! tee name=tee0
		! queue
		! multiratefirinterp name=multiratefirinterp0
		! lal_nofakedisconts silent=true
		! lal_checktimestamps
		! audio/x-raw-float,rate=%d
		! lal_nxydump
		! progressreport update-freq=1
		! filesink location=y.txt sync=no async=yes
		tee0.
		! queue
		! lal_nxydump
		! progressreport update-freq=1
		! filesink location=x.txt sync=no async=yes
""" % (inrate, outrate))

multiratefirinterp = pipeline.get_by_name("multiratefirinterp0")
multiratefirinterp.set_property("kernel", kernel)
multiratefirinterp.set_property("lag", num_zeros + 1)

pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
