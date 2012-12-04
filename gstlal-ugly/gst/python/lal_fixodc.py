# Copyright (C) 2012  Kipp Cannon
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


import pygtk
pygtk.require("2.0")
import gobject
import pygst
pygst.require('0.10')
import gst


from gstlal import pipeio


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


class lal_fixodc(gst.BaseTransform):
	__gstdetails__ = (
		"Fix ODC sample format",
		"Generic",
		"Type-casts float to int",
		__author__
	)

	__gproperties__ = {
	}

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-int, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) 1, " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 32," +
				"depth = (int) 32," +
				"signed = (bool) false"
			)
		)
	)


	def __init__(self):
		super(lal_checktimestamps, self).__init__()
		self.set_gapaware(True)


	def do_transform_caps(self, direction, caps):
		if direction == gst.PAD_SRC:
			tmpltcaps = self.get_pad("sink").get_pad_template_caps()
		elif direction == gst.PAD_SINK:
			tmpltcaps = self.get_pad("src").get_pad_template_caps()
		else:
			raise AssertionError
		rate, = [s["rate"] for s in caps]
		result = gst.Caps()
		for s in tmpltcaps:
			s = s.copy()
			s["rate"] = rate
			result.append_structure(s)
		return result


	def do_transform(self, ibuf, obuf):
		pipeio.array_from_audio_buffer(obuf)[:] = pipeio.array_from_audio_buffer(ibuf)
		return gst.FLOW_OK


#
# register element class
#


gobject.type_register(lal_fixodc)

__gstelementfactory__ = (
	lal_fixodc.__name__,
	gst.RANK_NONE,
	lal_fixodc
)
