#
# Copyright (C) 2011  Kipp Cannon
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


"""
Stuff to help add an http control and query interface to a program.
"""


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import socket
import sys
import threading
from gstlal import bottle
from gstlal import servicediscovery


#
# =============================================================================
#
#                            HTTP Interface Helpers
#
# =============================================================================
#


class HTTPServers(list):
	"""
	Utility to start, advertise, track and shutdown http servers on all
	interfaces.  De-advertise and shutdown the servers by deleting this
	object.  Do not allow the object to be garbage collected until you
	wish the servers to be shutdown.

	Example:

	>>> # save return value in a variable to prevent garbage collection
	>>> servers = HTTPServers(12345)
	>>> pass	# blah
	>>> pass	# blah
	>>> pass	# blah
	>>> # shutdown servers by deleting object
	>>> del servers

	bottle_app should be a Bottle instance.  If bottle_app is None (the
	default) then the current default Bottle application is used.
	"""
	def __init__(self, port, bottle_app = None, service_name = "gstlal", verbose = False):
		if bottle_app is None:
			bottle_app = bottle.default_app()
		self.verbose = verbose
		self.service_publisher = servicediscovery.Publisher()
		service_name = "%s.%s" % (service_name, servicediscovery.DEFAULT_STYPE)
		for (ignored, ignored, ignored, ignored, (_host, _port)) in socket.getaddrinfo(None, port, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST | socket.AI_PASSIVE):
			httpd = bottle.WSGIRefServer(host = _host, port = _port)
			httpd_thread = threading.Thread(target = httpd.run, args = (bottle_app,))
			httpd_thread.daemon = True
			httpd_thread.start()
			self.append((httpd, httpd_thread))
			while httpd.port == 0:
				pass
			if verbose:
				print >>sys.stderr, "started http server on http://%s:%d" % (httpd.host, httpd.port)
			self.service_publisher.addservice(servicediscovery.ServiceInfo(
				servicediscovery.DEFAULT_STYPE,
				service_name,
				address = socket.inet_aton(httpd.host),
				port = httpd.port
			))
			if verbose:
				print >>sys.stderr, "advertised http server on http://%s:%d as service \"%s\"" % (httpd.host, httpd.port, service_name)
		if not self:
			raise ValueError("unable to start servers%s" % (" on port %d" % port if port != 0 else ""))

	def __del__(self):
		while self:
			httpd, httpd_thread = self.pop()
			if self.verbose:
				print >>sys.stderr, "stopping http server on http://%s:%d ..." % httpd.srv.server_address,
			try:
				httpd.srv.shutdown()
			except Exception as e:
				result = "failed: %s" % str(e)
			else:
				result = "done"
			httpd_thread.join()
			if verbose:
				print >>sys.stderr, result
		self.service_publisher.unpublish()
