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
del pygtk

import gobject

import pygst
pygst.require('0.10')
del pygst

import gst
from gst.extend.pygobject import gproperty, with_construct_properties


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
