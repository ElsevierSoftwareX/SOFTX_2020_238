#!/usr/bin/env python
"""
Demonstration of a fast wavelet transform.

TODO: Make bin into an element, clean up code.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"

from gstlal.pipeutil import *
from gstlal.lloidparts import mkelems_fast, LLOIDHandler
import numpy as np


# Discrete Meyer wavelet kernels genreated with the following MATLAB command:
# [lo_d, hi_d, lo_r, hi_r] = wfilters('dmey')

hi_d = np.array([1.5097e-06,1.2788e-06,-4.4959e-07,-2.0966e-06,-1.7232e-06,6.9808e-07,2.8794e-06,2.3831e-06,-9.8252e-07,-4.2178e-06,-3.3535e-06,1.6747e-06,6.0345e-06,4.8376e-06,-2.4023e-06,-9.5563e-06,-7.2165e-06,4.8491e-06,1.4207e-05,1.0504e-05,-6.1876e-06,-2.4438e-05,-2.0106e-05,1.4994e-05,4.6429e-05,3.2341e-05,-3.741e-05,-0.00010278,-2.4462e-05,0.00014971,7.5593e-05,-0.00013991,9.3513e-05,0.00016119,-0.0008595,-0.00057819,0.0027022,0.0021948,-0.0060455,-0.0063867,0.011045,0.015251,-0.017404,-0.032094,0.024322,0.063667,-0.030621,-0.1327,0.035048,0.4441,-0.74375,0.4441,0.035048,-0.1327,-0.030621,0.063667,0.024322,-0.032094,-0.017404,0.015251,0.011045,-0.0063867,-0.0060455,0.0021948,0.0027022,-0.00057819,-0.0008595,0.00016119,9.3513e-05,-0.00013991,7.5593e-05,0.00014971,-2.4462e-05,-0.00010278,-3.741e-05,3.2341e-05,4.6429e-05,1.4994e-05,-2.0106e-05,-2.4438e-05,-6.1876e-06,1.0504e-05,1.4207e-05,4.8491e-06,-7.2165e-06,-9.5563e-06,-2.4023e-06,4.8376e-06,6.0345e-06,1.6747e-06,-3.3535e-06,-4.2178e-06,-9.8252e-07,2.3831e-06,2.8794e-06,6.9808e-07,-1.7232e-06,-2.0966e-06,-4.4959e-07,1.2788e-06,1.5097e-06,0])
lo_d = np.array([0,-1.5097e-06,1.2788e-06,4.4959e-07,-2.0966e-06,1.7232e-06,6.9808e-07,-2.8794e-06,2.3831e-06,9.8252e-07,-4.2178e-06,3.3535e-06,1.6747e-06,-6.0345e-06,4.8376e-06,2.4023e-06,-9.5563e-06,7.2165e-06,4.8491e-06,-1.4207e-05,1.0504e-05,6.1876e-06,-2.4438e-05,2.0106e-05,1.4994e-05,-4.6429e-05,3.2341e-05,3.741e-05,-0.00010278,2.4462e-05,0.00014971,-7.5593e-05,-0.00013991,-9.3513e-05,0.00016119,0.0008595,-0.00057819,-0.0027022,0.0021948,0.0060455,-0.0063867,-0.011045,0.015251,0.017404,-0.032094,-0.024322,0.063667,0.030621,-0.1327,-0.035048,0.4441,0.74375,0.4441,-0.035048,-0.1327,0.030621,0.063667,-0.024322,-0.032094,0.017404,0.015251,-0.011045,-0.0063867,0.0060455,0.0021948,-0.0027022,-0.00057819,0.0008595,0.00016119,-9.3513e-05,-0.00013991,-7.5593e-05,0.00014971,2.4462e-05,-0.00010278,3.741e-05,3.2341e-05,-4.6429e-05,1.4994e-05,2.0106e-05,-2.4438e-05,6.1876e-06,1.0504e-05,-1.4207e-05,4.8491e-06,7.2165e-06,-9.5563e-06,2.4023e-06,4.8376e-06,-6.0345e-06,1.6747e-06,3.3535e-06,-4.2178e-06,9.8252e-07,2.3831e-06,-2.8794e-06,6.9808e-07,1.7232e-06,-2.0966e-06,4.4959e-07,1.2788e-06,-1.5097e-06])

def make_filter_pair(inrate):
	bin = gst.parse_bin_from_description("""
		capsfilter caps="audio/x-raw-float,rate=%(inrate)d" name=in ! tee name=t
			! queue name=highpass_queue
			! multiratefirdecim name=highpass_filter lag=%(lag)d
			! capsfilter caps="audio/x-raw-float,rate=%(outrate)d" name=highpass_capsfilter
		t.
			! queue name=lowpass_queue
			! multiratefirdecim name=lowpass_filter lag=%(lag)d
			! capsfilter caps="audio/x-raw-float,rate=%(outrate)d" name=lowpass_capsfilter
	""" % {'inrate': inrate, 'outrate': inrate / 2, 'lag': len(lo_d) / 2}, False)

	bin.get_by_name("highpass_filter").set_property("kernel", hi_d)
	bin.get_by_name("lowpass_filter").set_property("kernel", lo_d)

	bin.add_pad(gst.GhostPad("sink", bin.get_by_name("in").get_pad("sink")))
	bin.add_pad(gst.GhostPad("highpass", bin.get_by_name("highpass_capsfilter").get_pad("src")))
	bin.add_pad(gst.GhostPad("lowpass", bin.get_by_name("lowpass_capsfilter").get_pad("src")))

	return bin


inrate = 4096
pipeline = gst.Pipeline("wavelet")
mainloop = gobject.MainLoop()
handler = LLOIDHandler(mainloop, pipeline)

src = gst.element_factory_make("audiotestsrc")
src.set_property("wave", "pink-noise")
src.set_property("volume", 1.)
pipeline.add(src)

head = gst.element_factory_make("capsfilter")
head.set_property("caps", gst.Caps("audio/x-raw-float,rate=%d,channels=1,width=64" % inrate))
pipeline.add(head)
src.link(head)
head = head.get_pad("src")

mixer = gst.element_factory_make("videomixer")
mixer.set_property("background", "black")
pipeline.add(mixer)
fmpgc = gst.element_factory_make("ffmpegcolorspace")
pipeline.add(fmpgc)
imagesink = gst.element_factory_make("ximagesink")
pipeline.add(imagesink)
mixer.link(fmpgc)
fmpgc.link(imagesink)

for i in range(5):
	filter_pair = make_filter_pair(inrate * 2 ** -i)
	pipeline.add(filter_pair)
	head.link(filter_pair.get_pad("sink"))
	head = filter_pair.get_pad("lowpass")
	sink = gst.parse_bin_from_description("""
		cairovis_lineseries
			y-autoscale=no
			y-min=-.5
			y-max=.5
		! video/x-raw-rgb,width=500,height=144
		! videobox name=videobox0 top=%d border-alpha=0
	""" % (-144 * i), True)
	pipeline.add(sink)
	filter_pair.get_pad("highpass").link(sink.sink_pads().next())
	sink.link(mixer)

pipeline.set_state(gst.STATE_PLAYING)
mainloop.run()
