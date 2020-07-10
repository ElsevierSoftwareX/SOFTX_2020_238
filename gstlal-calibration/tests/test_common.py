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
#				   Preamble
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
from gstlal import datasource


GObject.threads_init()
Gst.init(None)


if sys.byteorder == "little":
	BYTE_ORDER = "LE"
else:
	BYTE_ORDER = "BE"


#
# =============================================================================
#
#				  Utilities
#
# =============================================================================
#


def complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, is_live = False, verbose = True, src_suffix = ""):
	head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, samplesperbuffer = int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=F%d%s, rate=%d, channels=2" % (width, BYTE_ORDER, rate))
	head = pipeparts.mktogglecomplex(pipeline, head)
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "src%s" % src_suffix)
	return head

def int_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, is_live = False, verbose = True):
	head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, samplesperbuffer = int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length), is_live = is_live)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=S%d%s, rate=%d, channels=%d" % (width, BYTE_ORDER, rate, channels))
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "src")
	return head

def test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, volume = 1, is_live = False, verbose = True, src_suffix = ""):
	if wave == "ligo":
		head = pipeparts.mkfakeLIGOsrc(pipeline, instrument = "H1", channel_name = "LSC-STRAIN")
	else:
		head = pipeparts.mkaudiotestsrc(pipeline, wave = wave, freq = freq, volume = volume, samplesperbuffer = int(buffer_length * rate), num_buffers = int(test_duration / buffer_length), is_live = is_live)
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=F%d%s, rate=%d, channels=%d, channel-mask=(bitmask)0x0" % (width, BYTE_ORDER, rate, channels))
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "src%s" % src_suffix)
	return head


def gapped_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, volume = 0.8, gap_frequency = None, gap_threshold = None, control_dump_filename = None, is_live = False, verbose = True):
	src = test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration, wave = wave, freq = freq, volume = volume, is_live = is_live, verbose = verbose)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw, format=F32%s, rate=%d, channels=1" % (BYTE_ORDER, rate))
	if control_dump_filename is not None:
		control = pipeparts.mktee(pipeline, control)
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = gap_threshold)

def gapped_int_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, channels = 1, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, is_live = False, verbose = True):
	src = int_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, channels = channels, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live, verbose = verbose)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw, format=F32%s, rate=%d, channels=1" % (BYTE_ORDER, rate))
	if control_dump_filename is not None:
		control = pipeparts.mktee(pipeline, control)
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mkgate(pipeline, src, control = control, threshold = gap_threshold)

def gapped_complex_test_src(pipeline, buffer_length = 1.0, rate = 2048, width = 64, test_duration = 10.0, wave = 5, freq = 0, gap_frequency = None, gap_threshold = None, control_dump_filename = None, tags = None, is_live = False, verbose = True):
	src = complex_test_src(pipeline, buffer_length = buffer_length, rate = rate, width = width, test_duration = test_duration, wave = wave, freq = freq, is_live = is_live, verbose = verbose)
	if tags is not None:
		src = pipeparts.mktaginject(pipeline, src, tags)
	if gap_frequency is None:
		return src
	control = pipeparts.mkcapsfilter(pipeline, pipeparts.mkaudiotestsrc(pipeline, wave = 0, freq = gap_frequency, blocksize = 8 * int(buffer_length * rate), volume = 1, num_buffers = int(test_duration / buffer_length)), "audio/x-raw, format=F32%s, rate=%d, channels=1" % (BYTE_ORDER, rate))
	if control_dump_filename is not None:
		control = pipeparts.mknxydumpsinktee(pipeline, pipeparts.mkqueue(pipeline, control), control_dump_filename)
		control = pipeparts.mkqueue(pipeline, control)
	return pipeparts.mktogglecomplex(pipeline, pipeparts.mkgate(pipeline, pipeparts.mktogglecomplex(pipeline, src), control = control, threshold = gap_threshold))


#
# =============================================================================
#
#			       Pipeline Builder
#
# =============================================================================
#


def build_and_run(pipelinefunc, name, segment = None, **pipelinefunc_kwargs):
	print("=== Running Test %s ===" % name)
	mainloop = GObject.MainLoop()
	pipeline = pipelinefunc(Gst.Pipeline(name = name), name, **pipelinefunc_kwargs)
	handler = simplehandler.Handler(mainloop, pipeline)
	if segment is not None:
		if pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline failed to enter READY state")
		datasource.pipeline_seek_for_gps(pipeline, segment[0].ns() / 1000000000, segment[1].ns() / 1000000000)
	if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
		raise RuntimeError("pipeline failed to enter PLAYING state")
	pipeparts.write_dump_dot(pipeline, "test_%s" % name, verbose = True)
	mainloop.run()


#
# =============================================================================
#
#			Push Arrays Through an Element
#
# =============================================================================
#


def transform_arrays(input_arrays, elemfunc, name, rate = 1, **elemfunc_kwargs):
	input_arrays = list(input_arrays)	# so we can modify it
	output_arrays = []

	pipeline = Gst.Pipeline(name = name)

	head = pipeparts.mkgeneric(pipeline, None, "appsrc", caps = pipeio.caps_from_array(input_arrays[0], rate = rate))
	def need_data(elem, arg, input_arrays, rate):
		if input_arrays:
			arr = input_arrays.pop(0)
			elem.set_property("caps", pipeio.caps_from_array(arr, rate))
			buf = pipeio.audio_buffer_from_array(arr, 0, 0, rate)
			elem.emit("push-buffer", pipeio.audio_buffer_from_array(arr, 0, 0, rate))
			return Gst.FlowReturn.OK
		else:
			elem.emit("end-of-stream")
			return Gst.FlowReturn.EOS
	head.connect("need-data", need_data, input_arrays, rate)

	head = elemfunc(pipeline, head, **elemfunc_kwargs)

	head = pipeparts.mkappsink(pipeline, head)
	def appsink_get_array(elem, output_arrays):
		output_arrays.append(pipeio.array_from_audio_sample(elem.emit("pull-sample")))
		return Gst.FlowReturn.OK

	head.connect("new-sample", appsink_get_array, output_arrays)
	build_and_run((lambda *args, **kwargs: pipeline), name)

	return output_arrays
