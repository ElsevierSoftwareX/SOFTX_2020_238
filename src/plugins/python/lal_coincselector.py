# Copyright (C) 2010 Leo Singer
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
"""
Select interesting coincidences based on estimated false alarm rates.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *


class lal_coincselector(gst.BaseTransform):
	__gstdetails__ = (
		'Coincidence Selector',
		'Generic',
		__doc__,
		__author__
	)

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				"channels = (int) [0, MAX]
			""")
		),
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				"channels = (int) [0, MAX]
			""")
		)
	)



# Register element class
gstlal_element_register(lal_coincselector)
