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

# FIXME: Pick which sngl_inspiral field to hijack.
# Currently I am using rsqveto_duration to store per-detector IFAR.

from gstlal.pipeutil import *
import pylal.xlal.datatypes.snglinspiraltable as sngl


class lal_coincselector(gst.BaseTransform):
	__gstdetails__ = (
		'Coincidence Selector',
		'Generic',
		__doc__,
		__author__
	)
	__gproperties__ = {
		'min-ifar': (
			gobject.TYPE_UINT64,
			'min-ifar',
			'Minimum network IFAR (inverse false alarm rate, also known as extrapolated waiting time) in nanoseconds.  Coincidences with an estimated IFAR less than this value will be dropped.',
			0, gobject.G_MAXULONG, 3600 * gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'dt': ( # FIXME: could the coinc and coincselector elements share this piece of information?
			gobject.TYPE_UINT64,
			'dt',
			'Coincidence window in nanoseconds.',
			0, gobject.G_MAXULONG, 3600 * gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
	}
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


	def do_transform(self, inbuf, outbuf):
		min_ifar = self.get_property('min-ifar')
		dt = float(self.get_property('dt'))
		rows = sngl.from_buffer(outbuf.data)
		stride = outbuf.caps[0]['channels']
		for sngl_group in [rows[i*stride:i*stride+stride] for i in range(len(rows) / stride)]:
			ifar = dt
			for row in sngl_group:
				if any(buffer(row)):
					ifar *= row.rsqveto_duration / dt
			


# Register element class
gstlal_element_register(lal_coincselector)
