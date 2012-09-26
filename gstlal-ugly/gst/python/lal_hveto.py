#!/opt/local/bin/python

import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from pylal.xlal.datatypes.snglburst import from_buffer as sngl_bursts_from_buffer
from gstlal import pipeio
from gstlal.pipeutil import gstlal_element_register

class lal_hveto(gst.BaseTransform):
	"""
	hveto doc
	"""

	__gstdetails__ = (
		"lal_hveto",
		"lal_hveto",
		"stuff",
		"Chris Pankow chris.pankow@ligo.org"
	)

	_srctemplate = gst.PadTemplate( 'src',
		gst.PAD_SRC,
		gst.PAD_ALWAYS,
		gst.caps_from_string( 
		"audio/x-raw-int, " +
		"rate = (int) [1, MAX], " +
		"channels = (int) [1, MAX], " +
		"endianness = (int) BYTE_ORDER, " +
		"width = (int) 64," +
		"depth = (int) 64," +
		"signed = (bool) {true, false}")
	)

	_sinktemplate = gst.PadTemplate( 'sink',
		gst.PAD_SINK,
		gst.PAD_ALWAYS,
		gst.caps_from_string( "application/x-lal-snglburst" )
	)

	__gsttemplates__ = ( _sinktemplate, _srctemplate )

	def __init__( self ):
		gst.BaseTransform.__init__(self)

	def do_start(self):
		self.trigs = []
		return True

	def do_transform(self, inbuf, outbuf):
		for row in sngl_bursts_from_buffer(inbuf):	
			#print row
			pass

		outbuf[:16384] = numpy.zeros( 16384, dtype=numpy.int64 ).data

		return gst.FLOW_OK

#gstlal_element_register(lal_hveto)
gobject.type_register(lal_hveto)

__gstelementfactory__ = (
    lal_hveto.__name__,
    gst.RANK_NONE,
    lal_hveto
)

