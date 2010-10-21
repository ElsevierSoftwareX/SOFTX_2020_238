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
			'Instrument name (e.g., "H1")',
			None,
			gobject.PARAM_WRITABLE
		),
		'channel-name': (
			gobject.TYPE_STRING,
			'channel-name',
			'Channel name (e.g., "LSC-STRAIN")',
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
				elem.set_property('samplesperbuffer', val / 8)
		elif prop.name in ('instrument', 'channel-name'):
			self.__tags[prop.name] = val
			tagstring = ','.join('%s="%s"' % kv for kv in self.__tags.iteritems())
			self.__taginject.set_property('tags', tagstring)

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


	def do_send_event(self, event):
		"""Override send_event so that SEEK events go straight to the source
		elements, bypassing the filter chains.  This makes it so that the
		entire bin can be SEEKed even before it is added to a pipeline."""
		if event.type == gst.EVENT_SEEK:
			success = True
			for elem in self.iterate_sources():
				success &= elem.send_event(event)
			return success
		else:
			return super(lal_fakeadvligosrc, self).send_event(event)


	def __init__(self):
		super(lal_fakeadvligosrc, self).__init__()

		self.__tags = {'units':'strain'}

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
			('taginject',)
		)

		for chain in chains:
			chain[-1].link(outputchain[0])
	
		self.__taginject = outputchain[-1]
		self.add_pad(gst.ghost_pad_new_from_template('src', outputchain[-1].get_static_pad('src'), self.__gsttemplates__[0]))



# Register element class
gstlal_element_register(lal_fakeadvligosrc)
