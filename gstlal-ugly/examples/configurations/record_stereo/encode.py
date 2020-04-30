#!/usr/bin/env python3
"""

Encode whitened, bandpassed h(t) from H1 and L1 as a stereo .wav file.

"""
__author__ = ("Leo Singer <leo.singer@ligo.org>", "Drew Keppel <drew.keppel@ligo.org>")


# Command line interface

from optparse import Option, OptionParser

opts, args = OptionParser(option_list = [
	Option("--gps-time", type="int", metavar="SECONDS", help="GPS time for trigger"),
]).parse_args()

if opts.gps_time is None:
	raise RuntimeError("Required argument --gps-time not provided")


# Pipeline

from gstlal.pipeutil import *
from gstlal.reference_psd import read_psd
from gstlal.lloidparts import *
from gstlal import simplehandler

pipeline = gst.Pipeline()
mainloop = gobject.MainLoop()
handler = simplehandler.Handler(mainloop, pipeline)

mid_time = long(round(opts.gps_time / 16.0) * 16)
start_time = mid_time - 16
end_time = mid_time + 16

seek = gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_NONE,
	gst.SEEK_TYPE_SET, start_time * gst.SECOND,
	gst.SEEK_TYPE_SET, end_time * gst.SECOND)

stereo = mkelems_fast(pipeline,
	"interleave",
	"audioamplify",
	{
		"amplification": 1.0e+17,
	},
	"wavenc",
	"progressreport",
	{
		"name": "progress_wav",
		"update-freq": 1,
	},
	"filesink",
	{
		"location": "stereo_H1_L1.wav",
		"sync": False,
		"async": False,
	})[0]

for ifo in ('H1', 'L1'):
	psd = read_psd('reference_psd.%s.xml.gz' % ifo)
	elems = mkelems_fast(pipeline,
		"lal_onlinehoftsrc",
		{
			"instrument": ifo,
		},
		"audioresample",
		{
			"quality": 9,
			"gap-aware": True,
		},
		"capsfilter",
		{
			"caps": gst.Caps("audio/x-raw-float,rate=4096"),
		},
		"progressreport",
		{
			"name": "progress_%s" % ifo,
			"update-freq": 1
		},
		"lal_whiten",
		{
			"fft-length": 8,
			"zero-pad": 0,
			"psd-mode": 1,
			"median-samples": 7,
			"average-samples": 64,
			"mean-psd": psd.data,
		},
		"audiochebband",
		{
			"lower-frequency": 45,
			"upper-frequency": 2000,
			"poles": 8,
		},
		"queue",
		stereo)
	elems[0].set_state(gst.STATE_READY)
	elems[0].send_event(seek)

pipeline.set_state(gst.STATE_PLAYING)
gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "encode")
mainloop.run()

