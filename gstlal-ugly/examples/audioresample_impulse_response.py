#!/usr/bin/env python3
"""

Tests impulse response of filter bank for various resampler configurations.

"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


# Command line interface

from optparse import OptionParser, Option
try: # FIXME Python 2.4 compatibility for cluster
	any
except:
	from glue.iterutils import any

opts, args = OptionParser(
	option_list = [
		Option("--bank", help="Name of an SVD pickle file."),
		Option("-d", "--downsample-quality", metavar="0..9", choices=[str(q) for q in range(10)]),
		Option("-u", "--upsample-quality", metavar="0..9", choices=[str(q) for q in range(10)]),
		Option("--output", metavar="FILE"),
	]
).parse_args()

if any(getattr(opts, key) is None for key in ('bank', 'downsample_quality', 'upsample_quality', 'output')):
	raise RuntimeError("Missing some required arguments")


# Pipeline

from gstlal.pipeutil import *
from gstlal import simplehandler
from gstlal import pipeparts
from gstlal.pipeutil import mkelems_fast
from gstlal.svd_bank import read_bank
import math

bank = read_bank(opts.bank)
upsample_quality = int(opts.upsample_quality)
downsample_quality = int(opts.downsample_quality)

pipeline = gst.Pipeline("impulse_response")
mainloop = gobject.MainLoop()
handler = simplehandler.Handler(mainloop, pipeline)

source_rate = max(bank.get_rates())

# Input stage

impulse_time = 874107198
injfile = "impulse_at_874107198.xml"

src_elems = mkelems_fast(pipeline,
	"audiotestsrc", {"wave": "silence"},
	"progressreport", {"name": "progress_src", "update-freq": 1},
	"capsfilter", {"caps": gst.Caps("audio/x-raw-float,width=64,channels=1,rate=%d" % source_rate)},
	"taginject", {"tags": "instrument=H1,channel-name=IMPULSE,units=strain"},
	"lal_simulation", {"xml-location": injfile},
	"tee",
)

snr_elems = mkelems_fast(pipeline,
	"lal_adder", {"sync": True},
	"capsfilter", {"caps": gst.Caps("audio/x-raw-float,rate=%d" % source_rate)},
	"progressreport", {"name": "progress_out", "update-freq": 1},
	"filesink", {"location": opts.output},
)

for bank_fragment in bank.bank_fragments:
	mkelems_fast(pipeline,
		src_elems[-1],
		"lal_delay", {"delay": int(round( (bank.filter_length - bank_fragment.end)*bank_fragment.rate ))},
		"queue", {"max-size-bytes": 0, "max-size-buffers": 0, "max-size-time": 4 * int(math.ceil(bank.filter_length)) * gst.SECOND},
		"audioamplify", {"clipping-method": 3, "amplification": 1/math.sqrt(pipeparts.audioresample_variance_gain(downsample_quality, source_rate, bank_fragment.rate))},
		"audioresample", {"quality": downsample_quality},
		"capsfilter", {"caps": gst.Caps("audio/x-raw-float,rate=%d" % bank_fragment.rate)},
		"lal_firbank", {"latency": -int(round(bank_fragment.start * bank_fragment.rate)) - 1, "fir-matrix": bank_fragment.orthogonal_template_bank},
		"lal_reblock",
		"lal_matrixmixer", {"matrix": bank_fragment.mix_matrix},
		"audioresample", {"quality": upsample_quality},
		snr_elems[0],
	)

# Seek to just before impulse; be paranoid and pad by 1.5 template lengths before and after.

src_elems[0].set_state(gst.STATE_READY)
src_elems[0].send_event(gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_NONE,
	gst.SEEK_TYPE_SET, long(round((impulse_time - 1.5 * bank.filter_length) * gst.SECOND)),
	gst.SEEK_TYPE_SET, long(round((impulse_time + 1.5 * bank.filter_length) * gst.SECOND))
))



pipeline.set_state(gst.STATE_PLAYING)
gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, "impulse_response")
mainloop.run()
