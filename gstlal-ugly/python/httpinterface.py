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
try:
	from gstlal import servicediscovery
except ImportError:
	# disable
	servicediscovery = None


#
# =============================================================================
#
#                            HTTP Interface Helpers
#
# =============================================================================
#


def my_ipv4_address():
	#
	# this is sort of ridiculous.  we open a udp socket to google's dns
	# server and check the ip address through which the connection was
	# made.  this only works if we have a functioning connection to the
	# public internet.  it also only works if the interface through
	# which outbound connections go is the one you want, e.g.,
	# on cluster head nodes this will not be the ip address facing the
	# private network, which is probably the one you really want.
	#

	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	try:
		s.connect(("8.8.8.8", 53))
	except socket.error:
		# didn't work.  hope the machine name is in a name server
		# that knows the right answer (otherwise we'll get whatever
		# is in /etc/hosts)
		return socket.gethostbyname(socket.getfqdn())
	else:
		return s.getsockname()[0]
	finally:
		s.close()


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
	def __init__(self, port, bottle_app = None, service_name = "gstlal", service_properties = None, verbose = False):
		if bottle_app is None:
			bottle_app = bottle.default_app()
		self.verbose = verbose
		if servicediscovery is not None:
			self.service_publisher = servicediscovery.Publisher()
		else:
			self.service_publisher = None
		for (ignored, ignored, ignored, ignored, (_host, _port)) in socket.getaddrinfo(None, port, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST | socket.AI_PASSIVE):
			httpd = bottle.WSGIRefServer(host = _host, port = _port)
			httpd_thread = threading.Thread(target = httpd.run, args = (bottle_app,))
			httpd_thread.daemon = True
			httpd_thread.start()
			self.append((httpd, httpd_thread))
			while httpd.port == 0:
				pass
			host = httpd.host if httpd.host != "0.0.0.0" else socket.getfqdn()
			if verbose:
				print >>sys.stderr, "started http server on http://%s:%d" % (host, httpd.port)
			if self.service_publisher is not None:
				if verbose:
					print >>sys.stderr, "advertising http server on http://%s:%d as service \"%s\" ..." % (host, httpd.port, service_name),
				try:
					self.service_publisher.addservice(
						stype = servicediscovery.DEFAULT_PROTO + servicediscovery.DEFAULT_DOMAIN,
						name = service_name,
						host = host,
						port = httpd.port,
						properties = service_properties
					)
				except Exception as e:
					if verbose:
						print >>sys.stderr, "failed: %s" % str(e)
				else:
					if verbose:
						print >>sys.stderr, "done"
			elif verbose:
				print >>sys.stderr, "service discovery not available, http server not advertised"
		if not self:
			raise ValueError("unable to start servers%s" % (" on port %d" % port if port != 0 else ""))

	def __del__(self):
		if self.service_publisher is not None:
			if self.verbose:
				print >>sys.stderr, "de-advertising http server(s) ...",
			try:
				self.service_publisher.unpublish()
			except Exception as e:
				if self.verbose:
					print >>sys.stderr, "failed: %s" % str(e)
			else:
				if self.verbose:
					print >>sys.stderr, "done"
		while self:
			httpd, httpd_thread = self.pop()
			if self.verbose:
				print >>sys.stderr, "stopping http server on http://%s:%d ..." % (httpd.host if httpd.host != "0.0.0.0" else socket.getfqdn(), httpd.port),
			try:
				httpd.shutdown()
			except Exception as e:
				result = "failed: %s" % str(e)
			else:
				result = "done"
			if self.verbose:
				print >>sys.stderr, result
				print >>sys.stderr, "killing http server thread ...",
			# wait 10 seconds, then give up
			httpd_thread.join(10.0)
			if self.verbose:
				print >>sys.stderr, "timeout" if httpd_thread.is_alive() else "done"
