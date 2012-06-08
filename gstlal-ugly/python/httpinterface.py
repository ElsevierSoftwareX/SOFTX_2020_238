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


import atexit
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


def start_servers(port, service_name = "gstlal", verbose = False):
	"""
	Utility to start http servers on all interfaces.  All servers are
	started listening on the given port.

	Returns a tuple of (server, thread) tuples, one for each http
	server started by the function.  The servers are
	bottle.WSGIRefServer instances, the threads are threading.Thread
	instances.  NOTE:  the return value should be considered an opaque
	object, we reserve the right to switch to a different server system
	in the future.

	httpd_stop() is registered as a Python atexit handler for each
	server started by this function, so it is not normally necessary to
	explicitly stop the servers;  they will be automatically shutdown
	when the application exits.
	"""
	servers_and_threads = []
	service_publisher = servicediscovery.Publisher()
	service_name = "%s.%s" % (service_name, servicediscovery.DEFAULT_STYPE)
	for (ignored, ignored, ignored, ignored, (host, port)) in socket.getaddrinfo(None, port, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST | socket.AI_PASSIVE):
		httpd = bottle.WSGIRefServer(host = host, port = port)
		httpd_thread = threading.Thread(target = lambda: httpd.run(bottle.default_app()))
		httpd_thread.daemon = True
		httpd_thread.start()
		atexit.register(httpd_stop, httpd, httpd_thread, verbose = verbose)
		servers_and_threads.append((httpd, httpd_thread))
		if verbose:
			print >>sys.stderr, "started http server on http://%s:%d" % (host, port)
		service_publisher.addservice(servicediscovery.ServiceInfo(
			servicediscovery.DEFAULT_STYPE,
			service_name,
			address = socket.inet_aton(host),
			port = port
		))
		if verbose:
			print >>sys.stderr, "advertised http server on http://%s:%d as service \"%s\"" % (host, port, service_name)
	if not servers_and_threads:
		raise ValueError("unable to start servers on port %d" % port)
	atexit.register(service_publisher.unpublish)
	return tuple(servers_and_threads)


def httpd_stop(httpd, httpd_thread, verbose = False):
	"""
	Utility to shutdown an http server and join the corresponding
	thread.  start_servers() will register this function as an atexit
	handler for each server it starts, so there is normally no need to
	explicitly call this function.
	"""
	if verbose:
		print >>sys.stderr, "stopping http server on http://%s:%d ..." % httpd.srv.server_address,
	try:
		httpd.srv.shutdown()
	except Exception, e:
		result = "failed: %s" % str(e)
	else:
		result = "done"
	httpd_thread.join()
	if verbose:
		print >>sys.stderr, result
