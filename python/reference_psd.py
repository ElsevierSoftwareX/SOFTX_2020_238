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


from glue.ligolw import ligolw
from glue.ligolw import utils
from glue.ligolw import param
from glue.ligolw import types as ligolw_types
from pylal import series as lalseries


from gstlal import pipeparts
from gstlal import lloidparts
from gstlal import pipeio


#
# =============================================================================
#
#                               PSD Measurement
#
# =============================================================================
#


def measure_psd(instrument, seekevent, detector, seg, rate, fake_data = None, online_data = False, injection_filename = None, psd_fft_length = 8, frame_segments = None, verbose = False):
	#
	# pipeline handler for PSD measurement
	#

	class PSDHandler(lloidparts.LLOIDHandler):
		def on_message(self, bus, message):
			if message.type == gst.MESSAGE_ELEMENT and message.structure.get_name() == "spectrum":
				self.psd = pipeio.parse_spectrum_message(message)
			else:
				super(type(self), self).on_message(bus, message)

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

	if verbose:
		print >>sys.stderr, "measuring PSD in segment %s" % str(seg)
		print >>sys.stderr, "building pipeline ..."
	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline("psd")
	handler = PSDHandler(mainloop, pipeline)

	head = lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = fake_data, online_data = online_data, injection_filename = injection_filename, frame_segments = frame_segments, verbose = verbose)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=[%d,MAX]" % rate)	# disallow upsampling
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % rate)
	head = pipeparts.mkqueue(pipeline, head, max_size_buffers = 8)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 0, zero_pad = 0, fft_length = psd_fft_length, average_samples = int(round(float(abs(seg)) / (psd_fft_length / 2) - 1)), median_samples = 7)
	pipeparts.mkfakesink(pipeline, head)

	#
	# process segment
	#

	if verbose:
		print >>sys.stderr, "putting pipeline into playing state ..."
	pipeline.set_state(gst.STATE_PLAYING)
	if verbose:
		print >>sys.stderr, "running pipeline ..."
	mainloop.run()

	#
	# done
	#

	if verbose:
		print >>sys.stderr, "PSD measurement complete"
	return handler.psd


def psd_instrument_dict(elem):
	out = {}
	for lw in elem.getElementsByTagName(u"LIGO_LW"):
		if not lw.hasAttribute(u"Name"):
			continue
		if lw.getAttribute(u"Name") != u"REAL8FrequencySeries":
			continue
		ifo = param.get_pyvalue(lw, u"instrument")
		out[ifo] = lalseries.parse_REAL8FrequencySeries(lw)
	return out


def read_psd(filename, verbose = False):
	return psd_instrument_dict(utils.load_filename(filename, verbose = verbose))


def write_psd(filename, psd, instrument=None, verbose = False):
	xmldoc = ligolw.Document()
	lw = xmldoc.appendChild(ligolw.LIGO_LW())
	fs = lw.appendChild(lalseries.build_REAL8FrequencySeries(psd))
	if instrument is not None:
		fs.appendChild(param.new_param('instrument', ligolw_types.FromPyType[str], instrument))
	utils.write_filename(xmldoc, filename, gz = (filename or "stdout").endswith(".gz"), verbose = verbose)
