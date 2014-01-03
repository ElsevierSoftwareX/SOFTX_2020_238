# Copyright (C) 2009--2011,2013, 2014  Kipp Cannon, Madeline Wade
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


import numpy
import sys


import pygtk
pygtk.require("2.0")
import gobject
import pygst
pygst.require("0.10")
import gst


from gstlal import pipeparts
from gstlal import simplehandler


gobject.threads_init()


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#

def test_src_ints(pipeline, buffer_length = 1.0, rate = 2048, width = 32, channels = 1, test_duration = 10.0):
	head = pipeparts.mkaudiotestsrc(pipeline, wave = 4, blocksize = 4 * int(rate * buffer_length), volume = 1, num_buffers = int(test_duration / buffer_length))
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=%d, rate=%d, channels=%d, signed=false" % (width, rate, channels))
	head = pipeparts.mkgeneric(pipeline, head, "exp")
	head = pipeparts.mkgeneric(pipeline, head, "lal_fixodc")
	return pipeparts.mkprogressreport(pipeline, head, "src")

def gapped_test_src_ints(pipeline, buffer_length = 1.0, rate = 2048, width = 32, channels = 1, test_duration = 10.0, gap_frequency = None, gap_threshold = None, control_dump_filename = None):
	src = test_src_ints(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 4 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=32, rate=%d, channels=1, signed=false" % rate)
	if control_dump_filename is not None:
		control = pipeparts.mktee(pipeline, control)
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = gap_threshold)

#
# =============================================================================
#
#                               Pipeline Builder
#
# =============================================================================
#


def build_and_run(pipelinefunc, name, segment = None, **pipelinefunc_kwargs):
	print >>sys.stderr, "=== Running Test %s ===" % name
	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline(name)
	handler = simplehandler.Handler(mainloop, pipelinefunc(pipeline, name, **pipelinefunc_kwargs))
	if segment is not None:
		if pipeline.set_state(gst.STATE_PAUSED) == gst.STATE_CHANGE_FAILURE:
			raise RuntimeError("pipeline failed to enter PLAYING state")
		pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, segment[0].ns(), gst.SEEK_TYPE_SET, segment[1].ns())
	if pipeline.set_state(gst.STATE_PLAYING) == gst.STATE_CHANGE_FAILURE:
		raise RuntimeError("pipeline failed to enter PLAYING state")
	mainloop.run()
