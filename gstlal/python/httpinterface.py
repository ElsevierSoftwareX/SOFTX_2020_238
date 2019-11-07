#
# Copyright (C) 2011,2012,2014,2016,2018  Kipp Cannon
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
import time
import warnings


from . import bottle
from . import servicediscovery

#
# =============================================================================
#
#                            HTTP Interface Helpers
#
# =============================================================================
#


class HTTPDServer(object):
	def __init__(self, host, port, bottle_app, verbose = False):
		self.host = host
		self.port = port
		self.bottle_app = bottle_app
		self.verbose = verbose

	def __enter__(self):
		self.httpd = bottle.WSGIRefServer(host = self.host, port = self.port)
		self.httpd_thread = threading.Thread(target = self.httpd.run, args = (self.bottle_app,))
		self.httpd_thread.daemon = True
		self.httpd_thread.start()
		if self.verbose:
			print >>sys.stderr, "waiting for http server to start ..."
		while self.httpd.port == 0:
			time.sleep(0.25)
		self.host = self.httpd.host
		self.port = self.httpd.port
		if self.verbose:
			print >>sys.stderr, "started http server on http://%s:%d" % (self.httpd.host, self.httpd.port)
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		if self.verbose:
			print >>sys.stderr, "stopping http server on http://%s:%d ..." % (self.httpd.host, self.httpd.port),
		try:
			self.httpd.shutdown()
		except Exception as e:
			result = "failed: %s" % str(e)
		else:
			result = "done"
		if self.verbose:
			print >>sys.stderr, result
			print >>sys.stderr, "killing http server thread ...",
		# wait 10 seconds, then give up
		self.httpd_thread.join(10.0)
		if self.verbose:
			print >>sys.stderr, "timeout" if self.httpd_thread.is_alive() else "done"


class HTTPServers(list):
	"""
	Utility to start, advertise, track and shutdown http servers on all
	interfaces.  De-advertise and shutdown the servers by deleting this
	object.  Do not allow the object to be garbage collected until you
	wish the servers to be shutdown.

	Example:

	>>> # save return value in a variable to prevent garbage collection
	>>> servers = HTTPServers(port = 12345)
	>>> pass	# blah
	>>> pass	# blah
	>>> pass	# blah
	>>> # shutdown servers by deleting object
	>>> del servers

	If port = 0 (the default) a port will be assigned randomly.
	bottle_app should be a Bottle instance.  If bottle_app is None (the
	default) then the current default Bottle application is used.
	"""
	def __init__(self, port = 0, bottle_app = None, service_name = "www", service_domain = None, service_properties = None, verbose = False, service_discovery = True):
		if bottle_app is None:
			bottle_app = bottle.default_app()
		self.verbose = verbose
		self.service_discovery = service_discovery
		if self.service_discovery:
			self.service_publisher = servicediscovery.Publisher().__enter__()
		else:
			warnings.warn("disabling service discovery, this web server won't be able to advertise the location of the services it provides.")
		for (ignored, ignored, ignored, ignored, (host, port)) in socket.getaddrinfo(None, port, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST | socket.AI_PASSIVE):
			httpd = HTTPDServer(host, port, bottle_app, verbose = verbose).__enter__()
			if verbose:
				print >>sys.stderr, "advertising http server \"%s\" on http://%s:%d ..." % (service_name, httpd.host, httpd.port),
			if self.service_discovery:
				service = self.service_publisher.add_service(
					sname = service_name,
					sdomain = service_domain,
					port = httpd.port,
					properties = service_properties,
					commit = False
				)
			else:
				service = None
			if verbose:
				print >>sys.stderr, "done (%s)" % (".".join((service.sname, service.sdomain)) if service else "")
			self.append((httpd, service))
		if not self:
			raise ValueError("unable to start servers%s" % (" on port %d" % port if port != 0 else ""))
		if self.service_discovery:
			self.service_publisher.commit()

	def __del__(self):
		if self.verbose:
			print >>sys.stderr, "de-advertising http server(s) ...",
		try:
			if self.service_discovery:
				self.service_publisher.__exit__(None, None, None)
		except Exception as e:
			if self.verbose:
				print >>sys.stderr, "failed: %s" % str(e)
		else:
			if self.verbose:
				print >>sys.stderr, "done"
		while self:
			try:
				self.pop()[0].__exit__(None, None, None)
			except Exception as e:
				if self.verbose:
					print >>sys.stderr, "failed: %s" % str(e)
