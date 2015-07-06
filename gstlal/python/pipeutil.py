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

## @file

## @package pipeutil
"""

Boilerplate code, shorthand, and utility functions for creating GStreamer
elements and pipelines.

"""
__author__       = "Leo Singer <leo.singer@ligo.org>"
__organization__ = ["LIGO", "California Institute of Technology"]
__copyright__    = "Copyright 2010, Leo Singer"
__all__          = ["gobject", "gst", "gstlal_element_register", "mkelem", "mkelems_in_bin", "splice"]


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


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



def splice(bin, pad, element):
	"""Splice element into an existing bin by teeing off an existing pad.

	If necessary, a tee is added to the pipeline in order to splice the new element.

	bin is an instance of gst.Bin or gst.Pipeline.  pad is a string that
	describes any pad inside that bin.  The syntax used in gst-launch is
	understood.  For example, the string 'foo.bar.bat' means the pad called 'bat'
	on the element called 'bar' in the bin called 'foo' inside bin.  'foo.bar.'
	refers to any pad on the element 'bar'.  element_or_pad is either an element
	or a pad.

	FIXME: implicit pad names not yet understood.
	"""

	padpath = pad.split('.')
	padname = padpath.pop()

	elem = bin
	for name in padpath:
		elem = elem.get_by_name(name)
		if elem is None:
			raise NameError("no such element: '%s'" % name)

	pad = elem.get_pad(padname)
	if pad is None:
		raise NameError("no such pad: '%s'" % padname)

	tee_type = gst.element_factory_find('tee').get_element_type()

	tee = pad.get_parent_element()
	if tee.__gtype__ != tee_type:
		peer_pad = pad.get_peer()
		if peer_pad is None:
			if hasattr(element, 'get_direction'):
				elem.get_pad('src').link(element)
			else:
				elem.link(element)
			return
		else:
			peer_element = peer_pad.get_parent_element()
			if peer_element.__gtype__ == tee_type:
				tee = peer_element
			else:
				if pad.get_direction() == gst.PAD_SINK:
					pad, peer_pad = peer_pad, pad
				pad.unlink(peer_pad)
				tee = gst.element_factory_make("tee")
				bin.add(tee)
				pad.link(tee.get_static_pad('sink'))
				tee.get_request_pad('src%d').link(peer_pad)
	if hasattr(element, 'get_direction'):
		tee.get_request_pad('src%d').link(element)
	else:
		tee.link(element)
