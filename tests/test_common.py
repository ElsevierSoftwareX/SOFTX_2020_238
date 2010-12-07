# Copyright (C) 2009,2010  Kipp Cannon
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


gobject.threads_init()


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def test_src(pipeline, buffer_length = 1.0, rate = 2048, test_duration = 10.0, wave = 5, freq = 0):
	head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length))
	return pipeparts.mkprogressreport(pipeline, pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=64, rate=%d" % rate), "src")


def gapped_test_src(pipeline, buffer_length = 1.0, rate = 2048, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None):
	src = test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = wave, freq = freq)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=64, rate=%d" % rate)
	if control_dump_filename is not None:
		control = pipeparts.mktee(pipeline, control)
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = gap_threshold)


#
# =============================================================================
#
#                               Pipeline Handler
#
# =============================================================================
#


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
			print >>sys.stderr, "error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()


def build_and_run(pipelinefunc, name, segment = None):
	print >>sys.stderr, "=== Running Test %s ===" % name
	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline(name)
	handler = Handler(mainloop, pipelinefunc(pipeline))
	if segment is not None:
		pipeline.set_state(gst.STATE_PAUSED)
		pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, segment[0].ns(), gst.SEEK_TYPE_SET, segment[1].ns())
	pipeline.set_state(gst.STATE_PLAYING)
	mainloop.run()
