# Copyright (C) 2012  Drew Keppel
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
Generate simulated Advanced Virgo h(t) based on the baseline document
https://wwwcascina.virgo.infn.it/advirgo/docs/AdV_refsens_100512.txt
"""
__author__ = "Drew Keppel <drew.keppel@ligo.org>"

from gstlal import pipeutil
from gstlal.pipeutil import gobject, gst
from math import cos, pi, sqrt


class lal_fakeadvvirgosrc(gst.Bin):

	__gstdetails__ = (
		'Fake Advanced Virgo Source',
		'Source',
		__doc__,
		__author__
	)

	__gproperties__ = {
		'blocksize': (
			gobject.TYPE_UINT64,
			'blocksize',
			'Number of samples in each outgoing buffer',
			0, gobject.G_MAXULONG, 16384 * 8 * 1, # min, max, default
			gobject.PARAM_WRITABLE
		),
		'instrument': (
			gobject.TYPE_STRING,
			'instrument',
			'Instrument name (e.g., "V1")',
			None,
			gobject.PARAM_WRITABLE
		),
		'channel-name': (
			gobject.TYPE_STRING,
			'channel-name',
			'Channel name (e.g., "h_16384Hz")',
			None,
			gobject.PARAM_WRITABLE
		)
	}

	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				audio/x-raw-float,
				channels = (int) 1,
				endianness = (int) BYTE_ORDER,
				width = (int) 64,
				rate = (int) 16384
			""")
		),
	)

	def do_set_property(self, prop, val):
		if prop.name == 'blocksize':
			# Set property on all sources
			for elem in self.iterate_sources():
				elem.set_property('blocksize', val)
		elif prop.name in ('instrument', 'channel-name'):
			self.__tags[prop.name] = val
			tagstring = ','.join('%s="%s"' % kv for kv in self.__tags.iteritems())
			self.__taginject.set_property('tags', tagstring)


	def do_send_event(self, event):
		"""
		Send SEEK and EOS events to the source elements, all others
		to the by default bins send all events to the sink
		elements.
		"""
		# FIXME:  seeks should go to sink elements as well
		success = True
		if event.type in (gst.EVENT_SEEK, gst.EVENT_EOS):
			for elem in self.iterate_sources():
				success &= elem.send_event(event)
		else:
			for elem in self.iterate_sinks():
				success &= elem.send_event(event)
		return success


	def __init__(self):
		super(lal_fakeadvvirgosrc, self).__init__()

		self.__tags = {'units':'strain'}

		chains = []

		# the f^-2 part of the spectrum
		peak_width = 1 - 1e-3
		vol = 2.6e-31
		a = [1.]
		b = [-1.0, 4 * peak_width, -6 * peak_width**2, 4 * peak_width**3, -peak_width**4]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': a, 'b': b}),
				))

		# this pole recreates a line at 16.83 Hz
		central_freq = cos(2*pi*16.85/16384)
		peak_width = 1. - 2e-5
		vol = 5e-28
		a = [1.]
		b = [-1.0, 2 * central_freq * peak_width, -peak_width**2]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': a, 'b': b}),
				))

		# this f^-1 response helps connect the f^-2 part to the bucket
		peak_width = 1. - 1e-3
		vol = 1.5e-27
		a = [1.]
		b = [-1.0, 2 * peak_width, -peak_width**2]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': a, 'b': b}),
				))

		# this broad pole models the bump in the bucket
		central_freq = cos(2*pi*50./16384)
		peak_width = 1. - 1.1e-1
		vol = 6.3e-26
		a = [1.]
		b = [-1.0, 2 * central_freq * peak_width, -peak_width**2]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': a, 'b': b}),
				))

		# this pole recreates a line at 438 Hz
		central_freq = cos(2*pi*438./16384)
		peak_width = 1. - 7e-5
		vol = 2e-27
		a = [1.]
		b = [-1.0, 2 * central_freq * peak_width, -peak_width**2]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': a, 'b': b}),
				))

		# this is a 1st order one-sided finite-difference second-derivative stencil
		# it has a response of |H(f)| \propto f^2
		# FIXME: this could be tuned to more closely match at high
		# frequencies the baseline found above
		vol = 1.8e-22
		ker = [1.0, -2.0, 1.0]
		chains.append(pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': vol, 'samplesperbuffer': 16384}),
				('audiofirfilter', {'kernel': ker}),
				))

		outputchain = pipeutil.mkelems_in_bin(self,
			('lal_adder', {'sync': True}),
			('audioamplify', {'clipping-method': 3, 'amplification': sqrt(16384.)*3/4}),
			('taginject',)
		)

		for chain in chains:
			chain[-1].link(outputchain[0])

		self.__taginject = outputchain[-1]
		self.add_pad(gst.ghost_pad_new_from_template('src', outputchain[-1].get_static_pad('src'), self.__gsttemplates__[0]))



# Register element class
gobject.type_register(lal_fakeadvvirgosrc)

__gstelementfactory__ = (
	lal_fakeadvvirgosrc.__name__,
	gst.RANK_NONE,
	lal_fakeadvvirgosrc
)
