#!/usr/bin/env python
__author__ = "Leo Singer <leo.singer@ligo.org>"


from optparse import Option, OptionParser
opts, args = OptionParser(
	option_list =
	[
		Option("--instrument", "-i", metavar="IFO", action="append", help="Instruments to analyze."),
		Option("--injections", metavar="FILE.xml", help="Injection filename (a la lalapps_inspinj)"),
		Option("--gps-start-time", "-s", metavar="INT", type="int", help="GPS time at which to start analysis."),
		Option("--gps-end-time", "-e", metavar="INT", type="int", help="GPS time at which to end analysis."),
		Option("--template-bank", metavar="FILE", help="Name of template bank file."),
		Option("--output", metavar="FILE.{xml,xml.gz,sqlite}", help="Name of output file.  If not provided, then the output stage will be omitted."),
	]
).parse_args()

if len(opts.instrument) == 0:
    raise ValueError, "require at least one instrument"
if (opts.gps_start_time is None) or (opts.gps_end_time is None):
    raise ValueError, "require start and end times"
if opts.template_bank is None:
    raise ValueError, "require template bank file"

opts.psd_fft_length = 8

from gstlal.gstlal_svd_bank import read_bank
from gstlal import lloidparts, pipeparts
from gstlal.pipeutil import gst, gobject
from gstlal.lloidparts import mkelems_fast
from gstlal import ligolw_output
from glue import segments
from pylal.datatypes import LIGOTimeGPS
import sys
import os


pipeline = gst.Pipeline()
mainloop = gobject.MainLoop()
handler = lloidparts.LLOIDHandler(mainloop, pipeline)


seekevent = gst.event_new_seek(
	1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT,
	gst.SEEK_TYPE_SET, opts.gps_start_time * gst.SECOND,
	gst.SEEK_TYPE_SET, opts.gps_end_time * gst.SECOND
)


coinc_elems = mkelems_fast(pipeline, "lal_coinc","progressreport", {"name": "progress_coinc"}, "tee")
clustered_coinc_elems = mkelems_fast(pipeline, coinc_elems[-1],
    "lal_coincselector", {"min-combined-eff-snr": 0, "min-waiting-time": 10000000000})
#skymap = mkelems_fast(pipeline, coinc, "lal_skymap", {"bank-filename": opts.template_bank})[-1]
#mkelems_fast(pipeline, skymap, "fakesink")
#mkelems_fast(pipeline, coinc, "fakesink")

if opts.output is not None:
	output_prefix=os.path.split(opts.output)[0]
	output_name=os.path.split(opts.output)[1]
	seg = segments.segment(LIGOTimeGPS(opts.gps_start_time), LIGOTimeGPS(opts.gps_end_time))
	data = {}
	data['all'] = ligolw_output.Data(opts.instrument, ligolw_output.make_process_params(opts), tmp_space=None, output=os.path.join(output_prefix,"".join(opts.instrument)+"-"+output_name), seg=seg, out_seg=seg, injections=opts.injections, comment="", verbose=True)
	data['clustered'] = ligolw_output.Data(opts.instrument, ligolw_output.make_process_params(opts), tmp_space=None, output=os.path.join(output_prefix,"".join(opts.instrument)+"-clustered_"+output_name), seg=seg, out_seg=seg, injections=opts.injections, comment="", verbose=True)
	for ifo in opts.instrument:
		data[ifo] = ligolw_output.Data([ifo], ligolw_output.make_process_params(opts), tmp_space=None, output=ifo+"-"+opts.output, seg=seg, out_seg=seg, injections=None, comment="", verbose=True)

	# NB: To mix XML and sqlite, must call prepare_output_file() after all
	# instances of Data have been initialized.
	data['all'].prepare_output_file()
	data['clustered'].prepare_output_file()
else:
	mkelems_fast(pipeline, coinc_elems[-1], "fakesink", {"sync": False, "async": False})
	mkelems_fast(pipeline, clustered_coinc_elems[-1], "fakesink", {"sync": False, "async": False})

for ifo in opts.instrument:
	bank = read_bank("bank.%s.pickle" % ifo)
	bank.logname = ifo # FIXME This is only need to give elements names, that should be automatic.
	rates = bank.get_rates()

	basicsrc = lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, ifo, None, online_data=True, injection_filename=opts.injections)
	basicsrc = mkelems_fast(pipeline, basicsrc, "progressreport", {"name": "progress_src_%s" % ifo})[-1]
	hoftdict = lloidparts.mkLLOIDsrc(pipeline, basicsrc, rates, psd_fft_length=opts.psd_fft_length)
	snr_tee = lloidparts.mkLLOIDhoftToSnr(pipeline, hoftdict, ifo, bank, lloidparts.mkcontrolsnksrc(pipeline, max(rates)))
	triggers = lloidparts.mkLLOIDsnrToTriggers(pipeline, snr_tee, bank, lal_triggergen_algorithm=2, lal_triggergen_max_gap=1.0)
	triggers = mkelems_fast(pipeline, triggers, "progressreport", {"name": "progress_trig_%s" % ifo})[-1]
	triggers_tee = mkelems_fast(pipeline, triggers, "tee")[-1]
	triggers_tee.link_pads("src%d", coinc_elems[0], "sink%d")
	# output a database for each detector
	if opts.output is not None:
		data[ifo].prepare_output_file()
		pipeparts.mkappsink(pipeline, triggers_tee).connect_after("new-buffer", lloidparts.appsink_new_buffer, data[ifo])
	#mkelems_fast(pipeline, snr_tee, "queue", skymap)

#
# output file
#

# FIXME make some of these kw args options
if opts.output is not None:
	pipeparts.mkappsink(pipeline, coinc_elems[-1]).connect_after("new-buffer", lloidparts.appsink_new_buffer, data['all'])
	pipeparts.mkappsink(pipeline, clustered_coinc_elems[-1]).connect_after("new-buffer", lloidparts.appsink_new_buffer, data['clustered'])

#
# Ready set go!
#

pipeline.set_state(gst.STATE_PLAYING)
gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_NON_DEFAULT_PARAMS, "network")
mainloop.run()

if opts.output is not None:
	for datafile in data.itervalues():
		datafile.write_output_file()
