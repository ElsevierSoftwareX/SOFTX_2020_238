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


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst


from gstlal import pipeparts
from gstlal import pipeio
from gstlal import simplehandler


GObject.threads_init()
Gst.init(None)


if sys.byteorder == "little":
	BYTE_ORDER = "LE"
else:
	BYTE_ORDER = "BE"


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, is_live = False, verbose = True):
	assert not width % 8
	samplesperbuffer = int(round(buffer_length * rate))
	head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, volume = 1, blocksize = (width / 8 * 2) * samplesperbuffer, samplesperbuffer = samplesperbuffer, num_buffers = int(round(test_duration / buffer_length)), is_live = is_live)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=Z%d%s, rate=%d, channels=2" % (width, BYTE_ORDER, rate))
	head = pipeparts.mktogglecomplex(pipeline, head)
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "src")
	return head


def test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, is_live = False, verbose = True):
	assert not width % 8
	if wave == "ligo":
		head = pipeparts.mkfakeLIGOsrc(pipeline, instrument = "H1", channel_name = "LSC-STRAIN")
	else:
		samplesperbuffer = int(round(buffer_length * rate))
		head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, volume = 1, blocksize = (width / 8 * channels) * samplesperbuffer, samplesperbuffer = samplesperbuffer, num_buffers = int(round(test_duration / buffer_length)), is_live = is_live)
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=F%d%s, rate=%d, channels=%d" % (width, BYTE_ORDER, rate, channels))
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "src")
	return head


def add_gaps(pipeline, head, buffer_length, rate, test_duration, gap_frequency = None, gap_threshold = None, control_dump_filename = None):
	if gap_frequency is None:
		return head
	samplesperbuffer = int(round(buffer_length * rate))
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, volume = 1, blocksize = 4 * samplesperbuffer, samplesperbuffer = samplesperbuffer, num_buffers = int(round(test_duration / buffer_length))), "audio/x-raw, format=F32%s, rate=%d, channels=1" % (BYTE_ORDER, rate))
	if control_dump_filename is not None:
		control = pipeparts.mknxydumpsinktee(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, head, control = control, threshold = gap_threshold)


def gapped_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, tags = None, is_live = False, verbose = True):
	src = test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live, verbose = verbose)
	if tags is not None:
		src = pipeparts.mktaginject(pipeline, src, tags)
	return add_gaps(pipeline, src, buffer_length = buffer_length, rate = rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename)


def gapped_complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, tags = None, is_live = False, verbose = True):
	src = complex_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live, verbose = verbose)
	if tags is not None:
		src = pipeparts.mktaginject(pipeline, src, tags)
	return pipeparts.mktogglecomplex(pipeline, add_gaps(pipeline, pipeparts.mktogglecomplex(pipeline, src), buffer_length = buffer_length, rate = rate, test_duration = test_duration, gap_frequency = gap_frequency, gap_threshold = gap_threshold, control_dump_filename = control_dump_filename))


#
# =============================================================================
#
#                               Pipeline Builder
#
# =============================================================================
#


def build_and_run(pipelinefunc, name, segment = None, **pipelinefunc_kwargs):
	print("=== Running Test %s ===" % name, file=sys.stderr)
	mainloop = GObject.MainLoop()
	pipeline = pipelinefunc(Gst.Pipeline(name = name), name, **pipelinefunc_kwargs)
	handler = simplehandler.Handler(mainloop, pipeline)
	if segment is not None:
		if pipeline.set_state(Gst.State.PAUSED) == Gst.StateChangeReturn.FAILURE:
			raise RuntimeError("pipeline failed to enter PLAYING state")
		pipeline.seek(1.0, Gst.Format(Gst.Format.TIME), Gst.SeekFlags.FLUSH, Gst.SeekType.SET, segment[0].ns(), Gst.SeekType.SET, segment[1].ns())
	if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
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

	pipeline = Gst.Pipeline(name = name)

	head = pipeparts.mkgeneric(pipeline, None, "appsrc", caps = pipeio.caps_from_array(input_arrays[0], rate = rate))
	def need_data(elem, arg, input_array_rate_pair):
		input_arrays, rate = input_array_rate_pair
		if input_arrays:
			arr = input_arrays.pop(0)
			elem.set_property("caps", pipeio.caps_from_array(arr, rate))
			buf = pipeio.audio_buffer_from_array(arr, 0, 0, rate)
			elem.emit("push-buffer", pipeio.audio_buffer_from_array(arr, 0, 0, rate))
			return Gst.FlowReturn.OK
		else:
			elem.emit("end-of-stream")
			return Gst.FlowReturn.EOS
	head.connect("need-data", need_data, (input_arrays, rate))

	head = elemfunc(pipeline, head, **elemfunc_kwargs)

	head = pipeparts.mkappsink(pipeline, head)
	def appsink_get_array(elem, output_arrays):
		output_arrays.append(pipeio.array_from_audio_sample(elem.emit("pull-sample")))
		return Gst.FlowReturn.OK

	head.connect("new-sample", appsink_get_array, output_arrays)
	build_and_run((lambda *args, **kwargs: pipeline), name)

	return output_arrays
