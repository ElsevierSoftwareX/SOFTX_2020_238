# Copyright (C) 2009--2013  LIGO Scientific Collaboration
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
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from pylal import datatypes as laltypes


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
	if name in ("audio/x-raw-complex", "audio/x-raw-float", "audio/x-raw-int"):
		assert struct["width"] % 8 == 0
		return struct["channels"] * struct["width"] // 8
	elif name == "video/x-raw-rgb":
		assert struct["bpp"] % 8 == 0
		return struct["width"] * struct["height"] * struct["bpp"] // 8
	raise ValueError(caps)


def numpy_dtype_from_caps(caps):
	struct = caps[0]
	name = struct.get_name()
	if name == "audio/x-raw-float":
		assert struct["width"] % 8 == 0
		return "f%d" % (struct["width"] // 8)
	elif name == "audio/x-raw-int":
		assert struct["width"] % 8 == 0
		if struct["signed"]:
			return "i%d" % (struct["width"] // 8)
		else:
			return "u%d" % (struct["width"] // 8)
	elif name == "audio/x-raw-complex":
		assert struct["width"] % 8 == 0
		return "c%d" % (struct["width"] // 8)
	raise ValueError(name)


def caps_from_numpy_dtype(dtype):
	if dtype.char == 'f':
		caps = gst.Caps("audio/x-raw-float, width=32")
	elif dtype.char == 'd':
		caps = gst.Caps("audio/x-raw-float, width=64")
	elif dtype.char == 'b':
		caps = gst.Caps("audio/x-raw-int, width=8, signed=true")
	elif dtype.char == 'B':
		caps = gst.Caps("audio/x-raw-int, width=8, signed=false")
	elif dtype.char == 'h':
		caps = gst.Caps("audio/x-raw-int, width=16, signed=true")
	elif dtype.char == 'H':
		caps = gst.Caps("audio/x-raw-int, width=16, signed=false")
	elif dtype.char == 'i':
		caps = gst.Caps("audio/x-raw-int, width=32, signed=true")
	elif dtype.char == 'I':
		caps = gst.Caps("audio/x-raw-int, width=32, signed=false")
	elif dtype.char == 'l':
		caps = gst.Caps("audio/x-raw-int, width=64, signed=true")
	elif dtype.char == 'L':
		caps = gst.Caps("audio/x-raw-int, width=64, signed=false")
	else:
		raise ValueError(dtype)
	caps[0]["endianness"] = {
		"=": 1234 if sys.byteorder == "little" else 4321,
		"<": 1234,
		">": 4321
	}[dtype.byteorder]
	return caps


def caps_from_array(arr, rate = None):
	caps = caps_from_numpy_dtype(arr.dtype)
	if rate is not None:
		caps[0]["rate"] = rate
	caps[0]["channels"] = arr.shape[1]
	return caps


def array_from_audio_buffer(buf):
	channels = buf.caps[0]["channels"]
	# FIXME:  conditional is work-around for broken handling of
	# zero-length buffers in numpy < 1.7.  remove and use frombuffer()
	# unconditionally when we can rely on numpy >= 1.7
	if len(buf) > 0:
		a = numpy.frombuffer(buf, dtype = numpy_dtype_from_caps(buf.caps))
	else:
		a = numpy.array((), dtype = numpy_dtype_from_caps(buf.caps))
	a.shape = (len(a) // channels, channels)
	return a


def audio_buffer_from_array(arr, timestamp, offset, rate):
	buf = gst.Buffer(arr.data)
	buf.caps = caps_from_array(arr, rate = rate)
	buf.timestamp = timestamp
	buf.duration = gst.SECOND * arr.shape[0]
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
	return laltypes.REAL8FrequencySeries(
		name = "PSD",
		epoch = laltypes.LIGOTimeGPS(0, message.timestamp),
		f0 = 0.0,
		deltaF = message.structure["delta-f"],
		sampleUnits = laltypes.LALUnit(message.structure["sample-units"].strip()),
		data = numpy.array(message.structure["magnitude"])
	)


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
		sample_units = laltypes.LALUnit(taglist["units"].strip())
	else:
		sample_units = None
	return {
		"instrument": instrument,
		"channel-name": channel_name,
		"sample-units": sample_units
	}
