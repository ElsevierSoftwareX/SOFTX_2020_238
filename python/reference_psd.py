#!/usr/bin/env python
#
# Copyright (C) 2010  Kipp Cannon, Chad Hanna, Leo Singer
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import sys


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst


from gstlal import pipeutil
from gstlal import pipeparts
from gstlal import lloidparts
from gstlal import pipeio


from glue.ligolw import ligolw
from glue.ligolw import utils
from pylal import series as lalseries


#
# =============================================================================
#
#                               PSD Measurement
#
# =============================================================================
#


def measure_psd(instrument, seekevent, detector, seg, rate, fake_data=False, online_data=False, injection_filename=None, psd_fft_length=8, verbose=False):
	#
	# pipeline handler for PSD measurement
	#

	class PSDHandler(lloidparts.LLOIDHandler):
		def on_message(self, bus, message):
			if message.type == gst.MESSAGE_ELEMENT and message.structure.get_name() == "spectrum":
				self.psd = pipeio.parse_spectrum_message(message)
			else:
				super(PSDHandler, self).on_message(bus, message)

	#
	# 8 FFT-lengths is just a ball-parky estimate of how much data is
	# needed for a good PSD, this isn't a requirement of the code (the
	# code requires a minimum of 1)
	#

	if float(abs(seg)) < 8 * psd_fft_length:
		raise ValueError, "segment %s too short" % str(seg)

	#
	# build pipeline
	#

	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline("psd")
	handler = PSDHandler(mainloop, pipeline)

	lloidparts.mkelems_fast(pipeline,
		lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data=fake_data, online_data=online_data, injection_filename = injection_filename, verbose=verbose),
		"audioresample", {"quality": 9},
		"capsfilter", {"caps": gst.Caps("audio/x-raw-float, rate=%d" % rate)},
		"queue", {"max-size-buffers": 8},
		"lal_whiten", {"psd-mode": 0, "zero-pad": 0, "fft-length": psd_fft_length, "average-samples": int(round(float(abs(seg)) / (psd_fft_length / 2) - 1)), "median-samples": 7},
		"fakesink", {"sync": False, "async": False},
	)

	#
	# process segment
	#

	pipeline.set_state(gst.STATE_PLAYING)
	mainloop.run()

	#
	# done
	#

	return handler.psd


def read_psd(filename, verbose = False):
	return lalseries.parse_REAL8FrequencySeries(utils.load_filename(filename, gz = (filename or "stdin").endswith(".gz"), verbose = verbose))


def write_psd(filename, psd, verbose = False):
	xmldoc = ligolw.Document()
	xmldoc.appendChild(ligolw.LIGO_LW()).appendChild(lalseries.build_REAL8FrequencySeries(psd))
	utils.write_filename(xmldoc, filename, gz = (filename or "stdout").endswith(".gz"), verbose = verbose)
