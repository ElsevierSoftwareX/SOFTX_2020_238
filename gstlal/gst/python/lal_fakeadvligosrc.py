# Copyright (C) 2010  Leo Singer
# Copyright (C) 2010  Drew Keppel
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
Generate simulated Advanced LIGO h(t) based on ZERO DET, high power in
http://lhocds.ligo-wa.caltech.edu:8000/advligo/AdvLIGO_noise_curves
"""
__author__ = "Drew Keppel <drew.keppel@ligo.org>, Leo Singer <leo.singer@ligo.org>"

#
# The filter design is by Drew Keppel, who added the mkfakeadvLIGOsrc()
# function to pipeparts.py in Apr 2010 (git hash 232ee6bf97).  The code was
# subsequently ported to a stand-alone Python element by Leo Singer (git
# hash c79657d297).
#

from gstlal import pipeutil
from gstlal.pipeutil import gobject, gst
from math import cos, pi, sqrt


class lal_fakeadvligosrc(gst.Bin):

	__gstdetails__ = (
		'Fake Advanced LIGO Source',
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
		super(lal_fakeadvligosrc, self).__init__()

		self.__tags = {'units':'strain'}

		chains = (
			pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 4e-28, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (1.,), 'b': (-1., 2*cos(2*pi*9.103/16384)*0.99995, -1.*0.99995**2)}),
			),
			pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 1.1e-23, 'samplesperbuffer': 16384}),
				('audiofirfilter', {'kernel': (1., -1.)}),
			),
			pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 1e-27, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (1.,), 'b': (-1.0, 2 * 0.999, -.999**2)}),
			),
			pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 4e-26, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (1.,), 'b': (-1., 2*cos(2*pi*50./16384)*.87, -.87**2)}),
			),
			pipeutil.mkelems_in_bin(self,
				('audiotestsrc', {'wave':'gaussian-noise', 'volume': 6.5e-24, 'samplesperbuffer': 16384}),
				('audioiirfilter', {'a': (1.,), 'b': (-1., -2*.45, -.45**2)}),
			),
		)

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
gobject.type_register(lal_fakeadvligosrc)

__gstelementfactory__ = (
	lal_fakeadvligosrc.__name__,
	gst.RANK_NONE,
	lal_fakeadvligosrc
)
