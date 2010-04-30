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


__author__ = "Leo Singer <leo.singer@ligo.org>"
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
		'generate simulated enhanced LIGO h(t)',
		__author__
	)

	gproperty(
		gobject.TYPE_UINT64,
		'blocksize',
		'Number of samples in each outgoing buffer',
		0, gobject.G_MAXULONG, 16384 * 8 * 1, # min, max, default
		readable=True, writable=True
	)

	def do_set_property(self, prop, val):
		if prop.name == 'blocksize':
			for src in self.__srcs:
				src.set_property('samplesperbuffer', val / 8)
		else:
			super(lal_fakeligosrc, self).set_property(prop.name, val)

	def do_get_property(self, prop):
		if prop.name == 'blocksize':
			return self.__srcs[0].get_property('samplesperbuffer') * 8
		else:
			return super(lal_fakeligosrc, self).get_property(prop.name)

	def __init__(self):
		super(lal_fakeligosrc, self).__init__()

		# List to store source elements
		self.__srcs = []

		# Build first filter chain
		elems1 = [mkelem('lal_whitehoftsrc', {'volume': 5.03407936516e-17, 'samplesperbuffer': 16384})]
		for idx in range(14):
			elems1.append(mkelem('audioiirfilter', {'a': (1.87140685e-05, 3.74281370e-05, 1.87140685e-05), 'b': (1., 1.98861643, -0.98869215)}))

		# Build second filter chain
		elems2 = [mkelem('lal_whitehoftsrc', {'volume': 1.39238913312e-20, 'samplesperbuffer': 16384})]
		elems2.append(mkelem('audioiirfilter', {'a': (9.17933667e-07, 1.83586733e-06, 9.17933667e-07), 'b': (1., 1.99728828, -0.99729195)}))

		# Build third filter chain
		elems3 = [mkelem('lal_whitehoftsrc', {'volume': 2.16333076528e-23, 'samplesperbuffer': 16384})]

		# Build fourth filter chain
		elems4 = [
			mkelem('lal_whitehoftsrc', {'volume': 1.61077910675e-20, 'samplesperbuffer': 16384}),
			mkelem('audioiirfilter', {'a': (0.5591789, 0.5591789), 'b': (1., -0.1183578)}),
			mkelem('audioiirfilter', {'a': (0.03780506, -0.03780506), 'b': (1.0, -0.9243905)})]

		elem5 = [mkelem('lal_adder', {'sync': True}), mkelem('audioamplify', {'clipping-method': 3, 'amplification': 16384.**.5})]
		self.add_many(*elem5)
		gst.element_link_many(*elem5)

		# Add all elements to the bin
		for elems in (elems1, elems2, elems3, elems4):
			self.add_many(*elems)
			if len(elems) > 1:
				gst.element_link_many(*elems)
			elems[-1].link(elem5[0])
			self.__srcs.append(elems[0])

		self.add_pad(gst.GhostPad('src', elem5[-1].get_static_pad('src')))
	__init__ = with_construct_properties(__init__)



# Register element class
gobject.type_register(lal_fakeligosrc)
__gstelementfactory__ = ('lal_fakeligosrc', gst.RANK_NONE, lal_fakeligosrc)
