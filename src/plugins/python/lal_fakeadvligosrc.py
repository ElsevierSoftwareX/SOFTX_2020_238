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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from gstlal.pipeutil import *


__author__ = "Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


class lal_fakeadvligosrc(gst.Bin):

	__gstdetails__ = (
		'Fake Advanced LIGO Source',
		'Source',
		'generate simulated Advanced LIGO h(t) based on ZERO DET, high power in http://lhocds.ligo-wa.caltech.edu:8000/advligo/AdvLIGO_noise_curves',
		__author__
	)

	gproperty(
		gobject.TYPE_UINT64,
		'blocksize',
		'Number of samples in each outgoing buffer',
		0, gobject.G_MAXULONG, 16384 * 8 * 1, # min, max, default
		readable=True, writable=True
	)

	gproperty(
		gobject.TYPE_STRING,
		'instrument',
		'Instrument name (e.g., "H1")',
		None,
		readable=True, writable=True
	)

	gproperty(
		gobject.TYPE_STRING,
		'channel-name',
		'Channel name (e.g., "LSC-STRAIN")',
		None,
		readable=True, writable=True
	)

	def do_set_property(self, prop, val):
		if prop.name == 'blocksize':
			for src in self.__srcs:
				src.set_property('samplesperbuffer', val / 8)
		elif prop.name == 'instrument':
			self.__instrument = val
			self.__taginject = 'instrument=%s,channel-name=%s,units=strain' % (self.__instrument, self.__channel_name)
		elif prop.name == 'channel-name':
			self.__channel_name = val
			self.__taginject = 'instrument=%s,channel-name=%s,units=strain' % (self.__instrument, self.__channel_name)
		else:
			super(lal_fakeadvligosrc, self).set_property(prop.name, val)

	def do_get_property(self, prop):
		if prop.name == 'blocksize':
			return self.__srcs[0].get_property('samplesperbuffer') * 8
		elif prop.name == 'instrument':
			return self.__instrument
		elif prop.name == 'channel-name':
			return self.__channel_name
		else:
			return super(lal_fakeadvligosrc, self).get_property(prop.name)

	@with_construct_properties
	def __init__(self):
		super(lal_fakeadvligosrc, self).__init__()
		self.__srcs = []
		self.__channel_name = ''
		self.__instrument = ''

		chains = (
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 4e-18, 'samplesperbuffer': 16384}),
				*((('audioiirfilter', {'a': (1.951516E-6, 3.903032E-6, 1.951516E-6), 'b': (1., 1.9970651, -0.99707294)}),) * 20)
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 1.2e-20, 'samplesperbuffer': 16384}),
				*((('audioiirfilter', {'a': (6.686792E-7, 1.3373584E-6, 6.686792E-7), 'b': (1.0, 1.9982744, -0.9982772)}),) * 3)
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 4e-22, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (6.686792E-7, 1.3373584E-6, 6.686792E-7), 'b': (1.0, 1.9982744, -0.9982772)})
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 3.6e-24, 'samplesperbuffer': 16384})
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 3.12e-23, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (8.003242E-4, 8.003242E-4), 'b': (1.0, 0.99843043)})
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 5.36e-20, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (0.5591789, 0.5591789), 'b': (1., -0.1183578)}),
				('audioiirfilter', {'a': (4.2278392E-4, -4.2278392E-4), 'b': (1.0, -0.9992149)})
			)
		)

		outputchain = mkelems_in_bin(self,
			('lal_adder', {'sync': True}),
			('audioamplify', {'clipping-method': 3, 'amplification': 16384.**.5}),
			('capsfilter', {'caps': gst.Caps('audio/x-raw-float, width=64, rate=16384')}),
			('taginject',)
		)


		for chain in chains:
			chain[-1].link(outputchain[0])
			self.__srcs.append(chain[0])
	
		self.__taginject = outputchain[-1]
		self.add_pad(gst.GhostPad('src', outputchain[-1].get_static_pad('src')))



# Register element class
gstlal_element_register(lal_fakeadvligosrc)
