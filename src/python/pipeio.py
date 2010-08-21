# Copyright (C) 2009  LIGO Scientific Collaboration
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


from pylal import datatypes as laltypes
import pylal.xlal.datatypes.snglinspiraltable as sngl
from gstlal.pipeutil import gst


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


def repack_complex_array_to_real(input):
	"""
	Repack a complex-valued array into a real-valued array with twice
	as many columns.  Used to set complex arrays as values on elements
	that expose them as real-valued array properties (gobject doesn't
	understand complex numbers).  The return value is a view into the
	input array.
	"""
	# FIXME:  this function shouldn't exist, we should add complex
	# types to gobject
	if input.dtype.kind != "c":
		raise TypeError, input
	return input.view(dtype = numpy.dtype("f%d" % (input.dtype.itemsize / 2)))


def repack_real_array_to_complex(input):
	"""
	Repack a real-valued array into a complex-valued array with half as
	many columns.  Used to retrieve complex arrays from elements that
	expose them as real-valued array properties (gobject doesn't
	understand complex numbers).  The return value is a view into the
	input array.
	"""
	# FIXME:  this function shouldn't exist, we should add complex
	# types to gobject
	if input.dtype.kind != "f":
		raise TypeError, input
	return input.view(dtype = numpy.dtype("c%d" % (input.dtype.itemsize * 2)))


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
		return struct["channels"] * struct["width"] / 8
	elif name == "video/x-raw-rgb":
		return struct["width"] * struct["height"] * struct["bpp"] / 8
	raise ValueError, caps


def numpy_dtype_from_caps(caps):
	struct = caps[0]
	name = struct.get_name()
	if name == "audio/x-raw-float":
		return "f%d" % (struct["width"] / 8)
	elif name == "audio/x-raw-int":
		if struct["signed"]:
			return "i%d" % (struct["width"] / 8)
		else:
			return "s%d" % (struct["width"] / 8)
	elif name == "audio/x-raw-complex":
		return "c%d" % (struct["width"] / 8)
	raise ValueError, name


def array_from_audio_buffer(buf):
	channels = buf.caps[0]["channels"]
	a = numpy.frombuffer(buf, dtype = numpy_dtype_from_caps(buf.caps))
	return a.reshape((len(a) / channels, channels))


#
# =============================================================================
#
#                          SnglInspiralTable buffers
#
# =============================================================================
#


def net_ifar(ifars, dt):
	"""Compute net inverse false alarm rate (IFAR) from individual detector IFARs."""
	net_ifar = dt
	for ifar in ifars:
		net_ifar *= (ifar / dt)
	return net_ifar


def sngl_inspiral_is_non_nil(row):
	for c in buffer(row)[:-4]:
		if ord(c):
			return True
	return False


nil_sngl_buffer = buffer(sngl.SnglInspiralTable())


def sngl_inspiral_groups_from_buffer(buf):
	"""Extract (possibly multi-channel) SnglInspiralTable records from a buffer."""
	if buf.size < len(nil_sngl_buffer):
		return
	rows = sngl.from_buffer(buf)
	caps = buf.caps[0]
	if 'channels' in caps.keys():
		stride = caps['channels']
	else:
		stride = 1
	for i in range(len(rows) / stride):
		yield tuple(row for row in rows[i*stride:i*stride+stride] if sngl_inspiral_is_non_nil(row))


def sngl_inspiral_groups_to_buffer(groups, stride):
	"""Convert (possibly multi-channel) SnglInspiralTable records to a buffer."""
	data = ''
	for i_group, group in enumerate(groups):
		for i_row, row in enumerate(group):
			data += str(buffer(row))
		for i_row in range(len(group), stride):
			data += str(nil_sngl_buffer)
	buf = gst.buffer_new_and_alloc(len(data))
	if len(data) > 0:
		buf[0:len(data)] = data
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
		epoch = laltypes.LIGOTimeGPS(0, message.structure["timestamp"]),
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
