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


def net_ifar(ifars, dt):
	"""Compute net inverse false alarm rate (IFAR) from individual detector IFARs."""
	net_ifar = dt
	for ifar in ifars:
		net_ifar *= (ifar / dt)
	return net_ifar


def sngl_inspiral_is_nil(row):
	for c in buffer(row)[:-4]:
		if ord(c):
			return False
	return True


def sngl_inspiral_groups_from_buffer(buf):
	"""Extract (possibly multi-channel) SnglInspiralTable records from a buffer."""
	rows = sngl.from_buffer(buf)
	caps = buf.caps[0]
	if caps.hasattr('channels'):
		stride = caps['channels']
	else:
		stride = 1
	for i in range(len(rows) / nchannels):
		yield tuple(row for row in rows[i*stride:i*stride+stride] if sngl_inspiral_is_nil(row))


nil_sngl_buffer = buffer(sngl.SnglInspiralTable())


def sngl_inspiral_groups_to_buffer(buf, groups):
	"""Convert (possibly multi-channel) SnglInspiralTable to a buffer."""
	caps = buf.caps[0]
	if caps.hasattr('channels'):
		stride = caps['channels']
	else:
		stride = 1
	ngroups = 0
	for i_group, group in enumerate(row, groups):
		ngroups += 1
		for i_row, row in enumerate(group):
			data = buffer(row)
			buf.data[(i_group * stride + i_row) * len(data):(i_group * stride + i_row + 1) * len(data)] = data
		for i_row in range(len(group), stride):
			buf.data[(i_group * stride + i_row) * len(data):(i_group * stride + i_row + 1) * len(data)] = nil_sngl_buffer
	buf.size = ngroups * stride * len(nil_sngl_buffer)


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
			0, gst.CLOCK_TIME_NONE, 3600 * gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'dt': ( # FIXME: could the coinc and coincselector elements share this piece of information?
			gobject.TYPE_UINT64,
			'dt',
			'Coincidence window in nanoseconds.',
			0, gst.CLOCK_TIME_NONE, 50 * gst.MSECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
	}
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		),
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		)
	)


	def do_transform(self, inbuf, outbuf):
		min_ifar = self.get_property('min-ifar')
		dt = float(self.get_property('dt'))
		sngl_inspiral_groups_to_buffer(outbuf,
			(group for group in sngl_inspiral_groups_from_buffer(inbuf)
			if net_ifar((row.rsqveto_duration for row in sngl_group), dt) >= min_ifar))
		return gst.FLOW_OK


# Register element class
gstlal_element_register(lal_coincselector)
