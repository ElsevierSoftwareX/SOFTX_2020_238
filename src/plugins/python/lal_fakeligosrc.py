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


class lal_fakeligosrc(gst.Bin):

	__gstdetails__ = (
		'Fake LIGO Source',
		'Source',
		'generate simulated initial LIGO h(t)',
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
		readable=False, writable=True
	)

	gproperty(
		gobject.TYPE_STRING,
		'channel-name',
		'Channel name (e.g., "LSC-STRAIN")',
		None,
		readable=False, writable=True
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
		else:
			super(lal_fakeligosrc, self).set_property(prop.name, val)


	def do_get_property(self, prop):
		if prop.name == 'blocksize':
			# Retrieve value of property from first source element
			return self.iterate_sources().next().get_property('samplesperbuffer') * 8
		else:
			return super(lal_fakeligosrc, self).get_property(prop.name)


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
			return super(lal_fakeligosrc, self).send_event(event)


	@with_construct_properties
	def __init__(self):
		super(lal_fakeligosrc, self).__init__()

		self.__tags = {'units':'strain'}

		# Build first filter chain
		chains = (
			mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 5.03407936516e-17, 'samplesperbuffer': 16384}),
				*((('audioiirfilter', {'a': (1.87140685e-05, 3.74281370e-05, 1.87140685e-05), 'b': (1., 1.98861643, -0.98869215)}),) * 14)
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave': 'gaussian-noise', 'volume': 1.39238913312e-20, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (9.17933667e-07, 1.83586733e-06, 9.17933667e-07), 'b': (1., 1.99728828, -0.99729195)})
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave': 'gaussian-noise', 'volume': 2.16333076528e-23, 'samplesperbuffer': 16384})
			),
			mkelems_in_bin(self,
				('audiotestsrc', {'wave': 'gaussian-noise', 'volume': 1.61077910675e-20, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (0.5591789, 0.5591789), 'b': (1., -0.1183578)}),
				('audioiirfilter', {'a': (0.03780506, -0.03780506), 'b': (1.0, -0.9243905)})
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

		self.__taginject = outputchain[-1]
		self.add_pad(gst.GhostPad('src', outputchain[-1].get_static_pad('src')))



# Register element class
gstlal_element_register(lal_fakeligosrc)
