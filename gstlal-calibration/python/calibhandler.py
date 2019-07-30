# Copyright (C) 2019  Maddie Wade, Aaron Viets
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

import numpy
import StringIO
import threading

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

from gstlal import simplehandler
from lal import LIGOTimeGPS

#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


		

#
# =============================================================================
#
#                               Pipeline Handler
#
# =============================================================================
#

class Handler(simplehandler.Handler):
	"""!
	A subclass of simplehandler.Handler to be used with e.g.,
 	gstlal_calibration

	Implements...
	"""
	def __init__(self, mainloop, pipeline, kafka_server = None, verbose = False):
		super(Handler, self).__init__(mainloop, pipeline)
		#
		# initialize
		#
		self.lock = threading.Lock()
		self.pipeline = pipeline
		self.verbose = verbose
		self.kafka_server = kafka_server
		if self.kafka_server is not None:
			from kafka import KafkaProducer
			self.producer = KafkaProducer(
					bootstrap_servers = [kafka_server],
					key_serializer = lambda m: json.dumps(m).encode('utf-8'),
					value_serializer = lambda m: json.dumps(m).encode('utf-8'),
				)

	def appsink_statevector_new_buffer(self, elem, ifo, bitmaskdict):
		with self.lock:
			# retrieve data from appsink buffer
			buf = elem.emit("pull-sample").get_buffer()
			result, mapinfo = buf.map(Gst.MapFlags.READ)
			buf_timestamp = LIGOTimeGPS(0, buf.pts)
			if mapinfo.data:
				s = StringIO.StringIO(mapinfo.data)
				time, state = s.getvalue().split('\n')[0].split()
				state = int(state)
				buf.unmap(mapinfo)
				monitor_dict = {}
				for key, bitmask in bitmaskdict.items():
					monitor_dict[key] = state & bitmask
				if self.kafka_server is not None:
					self.producer.send("%s_statevector_bit_check" % ifo, value = monitor_dict) 
			return Gst.FlowReturn.OK	

