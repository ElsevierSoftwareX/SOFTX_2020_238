# Copyright (C) 2009--2011,2013  Kipp Cannon
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
from gstlal import pipeio
from gstlal import simplehandler


gobject.threads_init()


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, is_live = False):
	head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=%d, rate=%d, channels=2" % (width, rate))
	head = pipeparts.mktogglecomplex(pipeline, head)
	return pipeparts.mkprogressreport(pipeline, head, "src")


def test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, is_live = False):
	if wave == "ligo":
		head = pipeparts.mkfakeLIGOsrc(pipeline, instrument = "H1", channel_name = "LSC-STRAIN")
	else:
		head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=%d, rate=%d, channels=%d" % (width, rate, channels))
	return pipeparts.mkprogressreport(pipeline, head, "src")


def gapped_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, is_live = False):
	src = test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=32, rate=%d, channels=1" % rate)
	if control_dump_filename is not None:
		control = pipeparts.mktee(pipeline, control)
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = gap_threshold)


def gapped_complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, tags = None, is_live = False):
	src = complex_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live)
	if tags is not None:
		src = pipeparts.mktaginject(pipeline, src, tags)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw-float, width=32, rate=%d, channels=1" % rate)
	if control_dump_filename is not None:
		control = pipeparts.mknxydumpsinktee(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mktogglecomplex(pipeline, pipeparts.mkgate(pipeline, pipeparts.mktogglecomplex(pipeline, src), control = control, threshold = gap_threshold))


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
	pipeline = pipelinefunc(gst.Pipeline(name), name, **pipelinefunc_kwargs)
	handler = simplehandler.Handler(mainloop, pipeline)
	if segment is not None:
		if pipeline.set_state(gst.STATE_PAUSED) == gst.STATE_CHANGE_FAILURE:
			raise RuntimeError("pipeline failed to enter PLAYING state")
		pipeline.seek(1.0, gst.Format(gst.FORMAT_TIME), gst.SEEK_FLAG_FLUSH, gst.SEEK_TYPE_SET, segment[0].ns(), gst.SEEK_TYPE_SET, segment[1].ns())
	if pipeline.set_state(gst.STATE_PLAYING) == gst.STATE_CHANGE_FAILURE:
		raise RuntimeError("pipeline failed to enter PLAYING state")
	mainloop.run()


#
# =============================================================================
#
#                        Push Arrays Through an Element
#
# =============================================================================
#


def transform_arrays(input_arrays, elemfunc, name, rate = 1, **elemfunc_kwargs):
	input_arrays = list(input_arrays)	# so we can modify it
	output_arrays = []

	pipeline = gst.Pipeline(name)

	head = pipeparts.mkgeneric(pipeline, None, "appsrc", caps = pipeio.caps_from_array(input_arrays[0], rate = rate))
	def need_data(elem, arg, (input_arrays, rate)):
		if input_arrays:
			elem.emit("push-buffer", pipeio.audio_buffer_from_array(input_arrays.pop(0), 0, 0, rate))
		else:
			elem.emit("end-of-stream")
	head.connect("need-data", need_data, (input_arrays, rate))

	head = elemfunc(pipeline, head, **elemfunc_kwargs)

	head = pipeparts.mkappsink(pipeline, head)
	def appsink_get_array(elem, output_arrays):
		output_arrays.append(pipeio.array_from_audio_buffer(elem.get_last_buffer()))
	head.connect("new-buffer", appsink_get_array, output_arrays)

	build_and_run((lambda *args, **kwargs: pipeline), name)

	return output_arrays
