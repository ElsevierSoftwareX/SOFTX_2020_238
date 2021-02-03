# Copyright (C) 2009--2016  Kipp Cannon
# Copyright (C) 2016  Chad Hanna
# Copyright (C) 2016  Patrick Brockill
# Copyright (C) 2016  Sarah Caudill
# Copyright (C) 2015  Ryan Everett
# Copyright (C) 2010  Leo Singer
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

## @file

## @package pipeio

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
gi.require_version('GstAudio', '1.0')
from gi.repository import GObject
from gi.repository import Gst
from gi.repository import GstAudio
GObject.threads_init()
Gst.init(None)


import lal


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                  Properties
#
# =============================================================================
#


def repack_complex_array_to_real(arr):
	"""
	Repack a complex-valued array into a real-valued array with twice
	as many columns.  Used to set complex arrays as values on elements
	that expose them as real-valued array properties (gobject doesn't
	understand complex numbers).  The return value is a view into the
	input array.
	"""
	# FIXME:  this function shouldn't exist, we should add complex
	# types to gobject
	if arr.dtype.kind != "c":
		raise TypeError(arr)
	assert arr.dtype.itemsize % 2 == 0
	return arr.view(dtype = numpy.dtype("f%d" % (arr.dtype.itemsize // 2)))


def repack_real_array_to_complex(arr):
	"""
	Repack a real-valued array into a complex-valued array with half as
	many columns.  Used to retrieve complex arrays from elements that
	expose them as real-valued array properties (gobject doesn't
	understand complex numbers).  The return value is a view into the
	input array.
	"""
	# FIXME:  this function shouldn't exist, we should add complex
	# types to gobject
	if arr.dtype.kind != "f":
		raise TypeError(arr)
	return arr.view(dtype = numpy.dtype("c%d" % (arr.dtype.itemsize * 2)))


#
# =============================================================================
#
#                                   Buffers
#
# =============================================================================
#


def get_unit_size(caps):
	struct = caps[0]
	name = struct.get_name()
	if name == "audio/x-raw":
		info = GstAudio.AudioInfo()
		info.from_caps(caps)
		return info.bpf
	elif name == "video/x-raw" and struct["format"] in ("RGB", "RGBA", "ARGB", "ABGR"):
		return struct["width"] * struct["height"] * (3 if struct["format"] == "RGB" else 4)
	raise ValueError(caps)


def numpy_dtype_from_caps(caps):
	formats_dict = {
		GstAudio.AudioFormat.F32: numpy.dtype("float32"),
		GstAudio.AudioFormat.F64: numpy.dtype("float64"),
		GstAudio.AudioFormat.S8: numpy.dtype("int8"),
		GstAudio.AudioFormat.U8: numpy.dtype("uint8"),
		GstAudio.AudioFormat.S16: numpy.dtype("int16"),
		GstAudio.AudioFormat.U16: numpy.dtype("uint16"),
		GstAudio.AudioFormat.S32: numpy.dtype("int32"),
		GstAudio.AudioFormat.U32: numpy.dtype("uint32")
	}

	custom_formats_dict = {
		"Z64LE" : numpy.dtype("complex64"),
		"Z128LE": numpy.dtype("complex128")
	}

	info = GstAudio.AudioInfo()
	info.from_caps(caps)

	if info.finfo.format in formats_dict:
		return formats_dict[info.finfo.format]
	elif caps.get_structure(0).get_string("format") in custom_formats_dict:
		return custom_formats_dict[caps.get_structure(0).get_string("format")]
	else:
		raise ValueError("unknown GstAudioFormat : %s" % caps.get_structure(0).get_string("format"))


def format_string_from_numpy_dtype(dtype, formats_dict = {
		numpy.dtype("float32"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F32),
		numpy.dtype("float64"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F64),
		numpy.dtype("int8"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.S8),
		numpy.dtype("uint8"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.U8),
		numpy.dtype("int16"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.S16),
		numpy.dtype("uint16"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.U16),
		numpy.dtype("int32"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.S32),
		numpy.dtype("uint32"): GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.U32),
		numpy.dtype("complex64") : "Z64LE",
		numpy.dtype("complex128") : "Z128LE"
	}):
	return formats_dict[dtype]


def caps_from_array(arr, rate = None):
	return Gst.Caps.from_string("audio/x-raw, format=(string)%s, rate=(int)%d, channels=(int)%d, layout=(string)interleaved, channel-mask=(bitmask)0" % (format_string_from_numpy_dtype(arr.dtype), rate, arr.shape[1]))


def array_from_audio_sample(sample):
	caps = sample.get_caps()
	success, channels = caps.get_structure(0).get_int("channels")
	assert success

	buf = sample.get_buffer()
	success, mapinfo = buf.map(Gst.MapFlags.READ)
	assert success

	a = numpy.frombuffer(mapinfo.data, dtype = numpy_dtype_from_caps(caps))
	buf.unmap(mapinfo)
	a.shape = len(a) // channels, channels

	return a


def audio_buffer_from_array(arr, timestamp, offset, rate):
	buf = Gst.Buffer.new_wrapped(arr.tobytes())
	buf.pts = timestamp
	buf.duration = (Gst.SECOND * arr.shape[0] + rate // 2) // rate
	buf.offset = offset
	buf.offset_end = offset + arr.shape[0]
	return buf


#
# =============================================================================
#
#                                   Messages
#
# =============================================================================
#


def parse_spectrum_message(message):
	"""
	Parse a "spectrum" message from the lal_whiten element, return a
	LAL REAL8FrequencySeries containing the strain spectral density.
	"""
	s = message.get_structure()
	psd = lal.CreateREAL8FrequencySeries(
		name = s["instrument"] if s.has_field("instrument") else "",
		epoch = lal.LIGOTimeGPS(0, message.timestamp),
		f0 = 0.0,
		deltaF = s["delta-f"],
		sampleUnits = lal.Unit(s["sample-units"].strip()),
		length = len(s["magnitude"])
	)
	psd.data.data = numpy.array(s["magnitude"])
	return psd


#
# =============================================================================
#
#                                     Tags
#
# =============================================================================
#


def parse_framesrc_tags(taglist):
	try:
		instrument = taglist["instrument"]
	except KeyError:
		instrument = None
	try:
		channel_name = taglist["channel-name"]
	except KeyError:
		channel_name = None
	if "units" in taglist:
		sample_units = lal.Unit(taglist["units"].strip())
	else:
		sample_units = None
	return {
		"instrument": instrument,
		"channel-name": channel_name,
		"sample-units": sample_units
	}
