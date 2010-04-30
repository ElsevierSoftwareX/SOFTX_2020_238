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


import pygtk
pygtk.require("2.0")
import gobject
import pygst
pygst.require('0.10')
import gst
from gst.extend.pygobject import gproperty, with_construct_properties


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


def mkelem(elemname, props={}):
	elem = gst.element_factory_make(elemname)
	for (k, v) in props.iteritems():
		elem.set_property(k, v)
	return elem


class lal_whitehoftsrc(gst.Bin):
	gproperty(
		gobject.TYPE_UINT64,
		'samplesperbuffer',
		'Number of samples in each outgoing buffer',
		0, # min
		2**64-1, # max
		16384, # default
		readable=True, writable=True
	)

	def do_set_property(self, prop, val):
		if prop.name == 'samplesperbuffer':
			self.__src.set_property(prop.name, val)
		else:
			super(lal_whitehoftsrc, self).set_property(prop.name, val)

	def do_get_property(self, prop):
		if prop.name == 'samplesperbuffer':
			return self.__src.get_property(prop.name)
		else:
			return super(lal_whitehoftsrc, self).get_property(prop.name)

	def __init__(self):
		gst.Bin.__init__(self)
		elems = (
			mkelem('audiotestsrc', {'wave': 9, 'samplesperbuffer': 16384}),
			mkelem('capsfilter', {'caps': gst.Caps('audio/x-raw-float, width=64, rate=16384')})
		)
		self.add_many(*elems)
		gst.element_link_many(*elems)
		elems[1].get_static_pad('sink')
		self.add_pad(gst.GhostPad('src', elems[1].get_static_pad('src')))
		self.__src = elems[0]
	__init__ = with_construct_properties(__init__)



# Register element class
gobject.type_register(lal_whitehoftsrc)
__gstelementfactory__ = ('lal_whitehoftsrc', gst.RANK_NONE, lal_whitehoftsrc)
