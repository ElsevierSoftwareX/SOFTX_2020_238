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
"""

Boilerplate code, shorthand, and utility functions for creating GStreamer
elements and pipelines.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__organization__ = ["LIGO", "California Institute of Technology"]
__copyright__    = "Copyright 2010, Leo Singer"


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

# Shouldn't need pygtk or pygst
del pygtk
del pygst


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


def gstlal_element_register(clazz):
	"""Class decorator for registering a Python element.  Note that decorator
	syntax was extended from functions to classes in Python 2.6, so until 2.6
	becomes the norm we have to invoke this as a function instead of by
	saying::

		@gstlal_element_register
		class foo(gst.Element):
			...
	
	Until then, you have to do::

		class foo(gst.Element):
			...
		gstlal_element_register(foo)
	"""
	from inspect import getmodule
	gobject.type_register(clazz)
	getmodule(clazz).__gstelementfactory__ = (clazz.__name__, gst.RANK_NONE, clazz)
	return clazz


def mkelem(elemname, props={}):
	"""Instantiate an element named elemname and optionally set some of its 
	properties from the dictionary props."""
	elem = gst.element_factory_make(elemname)
	for (k, v) in props.iteritems():
		elem.set_property(k, v)
	return elem


def mkelems_in_bin(bin, *pipedesc):
	"""Create an array of elements from a list of tuples, add them to a bin,
	link them sequentially, and return the list.  Example:
	
	mkelem(bin, ('audiotestsrc', {'wave':9}), ('audioresample',))
	
	is equivalent to
	
	audiotestsrc wave=9 ! audioresample
	"""
	elems = [mkelem(*elemdesc) for elemdesc in pipedesc]
	for elem in elems:
		bin.add(elem)
	if len(elems) > 1:
		gst.element_link_many(*elems)
	return elems
