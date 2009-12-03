# Copyright (C) 2009  Kipp Cannon
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


import gobject
import pygst
pygst.require('0.10')
import gst


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


class NoFakeDisconts(gst.BaseTransform):
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"ANY"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"ANY"
			)
		)
	)


	def __init__(self):
		gst.BaseTransform.__init__(self)
		self.set_passthrough(True)


	def do_start(self):
		self.next_offset = None
		self.next_timestamp = None
		return True


	def do_transform_ip(self, buf):
		if self.next_offset is not None:
			if buf.offset != self.next_offset or buf.timestamp != self.next_timestamp:
				buf.flag_set(gst.BUFFER_FLAG_DISCONT)
			else:
				buf.flag_unset(gst.BUFFER_FLAG_DISCONT)

		self.next_offset = buf.offset_end
		self.next_timestamp = buf.timestamp + buf.duration

		return gst.FLOW_OK


gobject.type_register(NoFakeDisconts)


def mknofakedisconts(pipeline, src):
	elem = NoFakeDisconts()
	pipeline.add(elem)
	src.link(elem)
	return elem
