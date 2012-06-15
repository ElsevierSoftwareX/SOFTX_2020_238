# Copyright (C) 2011 Madeline Wade
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os
import sys
import warnings
import copy
import math

import pygtk
pygtk.require('2.0')
import pygst
pygst.require('0.10')
from gstlal.option import OptionParser

from glue import segments
from glue import segmentsUtils
from pylal.datatypes import LIGOTimeGPS

#
# parse command line
#

parser = OptionParser()
parser.add_option("--frame-cache-H1", metavar = "fileanme", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).  This is required unless --fake-data or --online-data is used in which case it must not be set.")
parser.add_option("--frame-cache-H2", metavar = "fileanme", help = "Set the name of the LAL cache listing the LIGO-Virgo .gwf frame files (optional).  This is required unless --fake-data or --online-data is used in which case it must not be set.")
parser.add_option("--online-data", action = "store_true", help = "Use online DMT-STRAIN instead of a frame file (optional).")
parser.add_option("--fake-data", action = "store_true", help = "Instead of reading data from .gwf files, generate and process coloured Gaussian noise modelling the Initial LIGO design spectrum (optional).")
parser.add_option("--gps-start-time", metavar = "seconds", help = "Set the start time of the segment to analyze in GPS seconds (required).  Can be specified to nanosecond precision.")
parser.add_option("--gps-end-time", metavar = "seconds", help = "Set the end time of the segment to analyze in GPS seconds (required).  Can be specified to nanosecond precision.")
parser.add_option("--injections", metavar = "filename", help = "Set the name of the LIGO light-weight XML file from which to load injections (optional).")
parser.add_option("--channel-name", metavar = "name", default = "LSC-STRAIN", help = "Set the name of the channel to process (optional).  The default is \"LSC-STRAIN\".")
parser.add_option("-v", "--verbose", action = "store_true", help = "Be verbose (optional).")
parser.add_option("--reference-psd-H1", metavar = "filename", help = "PSD will be read from filename")
parser.add_option("--reference-psd-H2", metavar = "filename", help = "PSD will be read from filename")
parser.add_option("--write-psd-H1", metavar = "filename", help = "Write measured noise spectrum to this LIGO light-weight XML file (optional).  This option has no effect if --reference-psd is used.")
parser.add_option("--write-psd-H2",  metavar = "filename", help = "Write measured noise spectrum to this LIGO light-weight XML file (optional).  This option has no effect if --reference-psd is used.")
parser.add_option("--write-pipeline", metavar = "filename", help = "Write a DOT graph description of the as-built pipeline to this file (optional).  The environment variable GST_DEBUG_DUMP_DOT_DIR must be set for this option to work.")
options, filenames = parser.parse_args()

options.seg = segments.segment(LIGOTimeGPS(options.gps_start_time), LIGOTimeGPS(options.gps_end_time))
options.psd_fft_length = 8      # seconds
options.zero_pad_length = 2
rates = [16384, 2*2048]

#
# =============================================================================
#
#                                             Main
#
# =============================================================================
#

import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from gstlal import pipeutil
from gstlal import pipeparts
from gstlal import lloidparts
from gstlal import reference_psd

import pylal.xlal.datatypes.real8frequencyseries
import numpy

#
# set up source information
#

if options.gps_start_time is None:
          seek_start_type = gst.SEEK_TYPE_NONE
          seek_start_time = -1 # gst.CLOCK_TIME_NONE is exported as unsigned, should have been signed
else:
          seek_start_type = gst.SEEK_TYPE_SET
          seek_start_time = options.seg[0].ns()

if options.gps_end_time is None:
          seek_stop_type = gst.SEEK_TYPE_NONE
          seek_stop_time = -1 # gst.CLOCK_TIME_NONE is exported as unsigned, should have been signed
else:
          seek_stop_type = gst.SEEK_TYPE_SET
          seek_stop_time = options.seg[1].ns()

seekevent = gst.event_new_seek(1.0, gst.Format(gst.FORMAT_TIME),
          gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT,
          seek_start_type, seek_start_time,
          seek_stop_type, seek_stop_time)


detectorH1 = {
          "H1": lloidparts.DetectorData(options.frame_cache_H1, options.channel_name)
}

detectorH2 = {
          "H2": lloidparts.DetectorData(options.frame_cache_H2, options.channel_name)
}

#
# get H1 an H2 PSDs
#

if options.reference_psd_H1 is not None:
          PSD1 = reference_psd.read_psd(options.reference_psd_H1, verbose = options.verbose)
          PSD1, = PSD1.values()
else:
          PSD1 = reference_psd.measure_psd(
                    "H1",
                    seekevent,
                    detectorH1["H1"],
                    options.seg,
                    max(rates),       # Hz;  must not be less than highest bank fragment sample rate (see below)
                    psd_fft_length = options.psd_fft_length,
                    fake_data = options.fake_data,
                    online_data = options.online_data,
                    injection_filename = options.injections,
                    verbose = options.verbose
          )
          if options.write_psd_H1 is not None:
                    reference_psd.write_psd(options.write_psd_H1, PSD1, "H1", verbose = options.verbose)

if options.reference_psd_H2 is not None:
          PSD2 = reference_psd.read_psd(options.reference_psd_H2, verbose = options.verbose)
          PSD2, = PSD2.values()
else:
          PSD2 = reference_psd.measure_psd(
                    "H2",
                    seekevent,
                    detectorH2["H2"],
                    options.seg,
                    max(rates),       # Hz;  must not be less than highest bank fragment sample rate (see below)
                    psd_fft_length = options.psd_fft_length,
                    fake_data = options.fake_data,
                    online_data = options.online_data,
                    injection_filename = options.injections,
                    verbose = options.verbose
          )
          if options.write_psd_H2 is not None:
                    reference_psd.write_psd(options.write_psd_H2, PSD2, "H2", verbose = options.verbose)

#
# make the coherent PSD factors
#

ratio = PSD1.data*PSD2.deltaF/(PSD2.data*PSD1.deltaF)

PSD1_coh = PSD1
coh_factors_H1 = 1/(1+ratio)
PSD1_coh.data = 1/coh_factors_H1
PSD1_coh.data[0] = PSD1_coh.data[-1] = 0.0

PSD2_coh = PSD2
coh_factors_H2 = ratio*coh_factors_H1
PSD2_coh.data = 1/coh_factors_H2
PSD2_coh.data[0] = PSD2_coh.data[-1] = 0.0

#
# make the overwhitening PSDs
#

PSD1_coh.data = PSD1_coh.data*PSD1_coh.data
PSD2_coh.data = PSD2_coh.data*PSD2_coh.data

#
# take care of the extra factor of 2*df
#

for elem in PSD1_coh.data: 
	elem = elem*math.sqrt(2*PSD1_coh.deltaF)

for elem in PSD2_coh.data:
	elem = elem*math.sqrt(2*PSD2_coh.deltaF)

#
# function to deal with changing delta-f and f-nyquist
#

def psd_resolution_changed(elem, pspec, psd):
          # get frequency resolution
          f_nyquist = elem.get_property("f-nyquist")
          delta_f = elem.get_property("delta-f")
          #print delta_f, f_nyquist
          n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
          # interpolate and install PSD
          psd = reference_psd.interpolate_psd(psd, delta_f)
          elem.set_property("mean-psd", psd.data[:n])

#
# begin pipline
#

pipeline = gst.Pipeline("coherent_and_null_H1H2")
mainloop = gobject.MainLoop()
handler = lloidparts.LLOIDHandler(mainloop, pipeline)

#
# overwhiten the H1 data with coherent factors
#

H1src = lloidparts.mkLLOIDbasicsrc(
                    pipeline,
                    seekevent,
                    "H1",
                    detectorH1["H1"],
                    fake_data = options.fake_data,
                    injection_filename = options.injections,
                    verbose = options.verbose
          )

quality = 9
H1head = pipeparts.mkqueue(pipeline, H1src)
H1head = pipeparts.mkcapsfilter(pipeline, H1head, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
H1head = pipeparts.mkresample(pipeline, H1head, quality = quality)
H1head = pipeparts.mkcapsfilter(pipeline, H1head, "audio/x-raw-float, rate=%d" % max(rates))
H1tee = pipeparts.mktee(pipeline, H1head)
H1head = pipeparts.mkqueue(pipeline, H1tee)
H1head = pipeparts.mkwhiten(pipeline, H1head, fft_length = options.psd_fft_length, zero_pad = options.zero_pad_length, average_samples = 64, median_samples = 7)
H1head.set_property("psd-mode", 1)
H1head.connect_after("notify::f-nyquist", psd_resolution_changed, PSD1_coh)
H1head.connect_after("notify::delta-f", psd_resolution_changed, PSD1_coh)
H1head = pipeparts.mknofakedisconts(pipeline, H1head, silent = True)

#
# overwhiten the H2 data with coherent factors
#

H2src = lloidparts.mkLLOIDbasicsrc(
                    pipeline,
                    seekevent,
                    "H2",
                    detectorH2["H2"],
                    fake_data = options.fake_data,
                    injection_filename = options.injections,
                    verbose = options.verbose
          )

H2head = pipeparts.mkqueue(pipeline, H2src)
H2head = pipeparts.mkcapsfilter(pipeline, H2head, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
H2head = pipeparts.mkresample(pipeline, H2head, quality = quality)
H2head = pipeparts.mkcapsfilter(pipeline, H2head, "audio/x-raw-float, rate=%d" % max(rates))
H2tee = pipeparts.mktee(pipeline, H2head)
H2head = pipeparts.mkqueue(pipeline, H2tee)
H2head = pipeparts.mkwhiten(pipeline, H2head, fft_length = options.psd_fft_length, zero_pad = options.zero_pad_length, average_samples = 64, median_samples = 7)
H2head.set_property("psd-mode", 1)
H2head.connect_after("notify::f-nyquist", psd_resolution_changed, PSD2_coh)
H2head.connect_after("notify::delta-f", psd_resolution_changed, PSD2_coh)
H2head = pipeparts.mknofakedisconts(pipeline, H2head, silent = True)

#
# add the H1 and H2 overwhitened data streams
#

coh_adder = pipeutil.mkelem("lal_adder",{"sync": True})
pipeline.add(coh_adder)
H1head.link(coh_adder)
H2head.link(coh_adder)

COHhead = pipeparts.mkqueue(pipeline, coh_adder)
COHhead = pipeparts.mkprogressreport(pipeline, COHhead, "progress_coherent")
COHhead = pipeparts.mkframesink(pipeline, COHhead, clean_timestamps = False, dir_digits = 0, frame_type = "LHO_COHERENT")

#
# make the null stream
#

H1tee = pipeparts.mkqueue(pipeline, H1tee)

H2tee = pipeparts.mkqueue(pipeline, H2tee)
H2tee = pipeparts.mkaudioamplify(pipeline, H2tee, -1)
H2tee = pipeparts.mkqueue(pipeline, H2tee)

null_adder = pipeutil.mkelem("lal_adder", {"sync": True})
pipeline.add(null_adder)
H1tee.link(null_adder)
H2tee.link(null_adder)

NULLhead = pipeparts.mkqueue(pipeline, null_adder)
NULLhead = pipeparts.mkprogressreport(pipeline, NULLhead, "progress_null")
NULLhead = pipeparts.mkframesink(pipeline, NULLhead, clean_timestamps = False, dir_digits = 0, frame_type = "LHO_NULL")

#
# running the pipeline stuff
#

def write_dump_dot(pipeline, filestem, verbose = False):
          if "GST_DEBUG_DUMP_DOT_DIR" not in os.environ:
                    raise ValueError, "cannot write pipeline, environment variable GST_DEBUG_DUMP_DOT_DIR is not set"
          gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, filestem)
          if verbose:
                    print >>sys.stderr, "Wrote pipeline to %s" % os.path.join(os.environ["GST_DEBUG_DUMP_DOT_DIR"], "%s.dot" % filestem)

class Handler(object):
          def __init__(self, mainloop, pipeline):
                    self.mainloop = mainloop
                    self.pipeline = pipeline

                    bus = pipeline.get_bus()
                    bus.add_signal_watch()
                    bus.connect("message", self.on_message)

          def on_message(self, bus, message):
                    if message.type == gst.MESSAGE_EOS:
                              self.pipeline.set_state(gst.STATE_NULL)
                              self.mainloop.quit()
                    elif message.type == gst.MESSAGE_ERROR:
                              gerr, dbgmsg = message.parse_error()
                              self.pipeline.set_state(gst.STATE_NULL)
                              self.mainloop.quit()
                              sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))
                    elif message.type == gst.MESSAGE_STATE_CHANGED:
                              old,new,pending = message.parse_state_changed()
                              if new == gst.STATE_PLAYING:
                                        maybe_dump_dot(pipeline, "PLAYING")

if options.write_pipeline is not None:
          write_dump_dot(pipeline, "%s.%s" % (options.write_pipeline, "NULL"), verbose = options.verbose)

if options.verbose:
          print >>sys.stderr, "setting pipeline state to paused ..."
if pipeline.set_state(gst.STATE_PAUSED) != gst.STATE_CHANGE_SUCCESS:
          raise RuntimeError, "pipeline did not enter paused state"

if options.verbose:
          print >>sys.stderr, "setting pipeline state to playing ..."
if pipeline.set_state(gst.STATE_PLAYING) != gst.STATE_CHANGE_SUCCESS:
          raise RuntimeError, "pipeline did not enter playing state"

if options.write_pipeline is not None:
          write_dump_dot(pipeline, "%s.%s" % (options.write_pipeline, "PLAYING"), verbose = options.verbose)

if options.verbose:
          print >>sys.stderr, "running pipeline ..."
mainloop.run()
