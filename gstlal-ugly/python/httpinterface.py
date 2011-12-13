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
import BaseHTTPServer
import socket
import SocketServer
import sys
import threading


#
# =============================================================================
#
#                            HTTP Interface Helpers
#
# =============================================================================
#


def start_servers(port, handler_class, verbose = False):
	"""
	Utility to start http servers on all interfaces.  All servers are
	started listening on the given port, and will use the hander_class
	class to create handlers.  handler_class is typically a subclass of
	BaseHTTPServer.BaseHTTPRequestHandler.

	Returns a tuple of (server, thread) tuples, one for each http
	server started by the function.  The servers are
	SocketServer.ThreadingTCPServer instances, the threads are
	threading.Thread instances.

	httpd_stop() is registered as an atexit handler for each server
	started by this function.
	"""
	servers_and_threads = []
	for (ignored, ignored, ignored, ignored, sockaddr) in socket.getaddrinfo(None, port, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST | socket.AI_PASSIVE):
		httpd = SocketServer.ThreadingTCPServer(sockaddr, handler_class)
		httpd_thread = threading.Thread(target = httpd.serve_forever)
		httpd_thread.daemon = True
		httpd_thread.start()
		atexit.register(httpd_stop, httpd, httpd_thread, verbose = verbose)
		servers_and_threads.append((httpd, httpd_thread))
		if verbose:
			print >>sys.stderr, "started http server on %s" % repr(sockaddr)
	if not servers_and_threads:
		raise ValueError("unable to start servers on port %d" % port)
	return tuple(servers_and_threads)


def httpd_stop(httpd, httpd_thread, verbose = False):
	"""
	Utility to shutdown an http server and join the corresponding
	thread.  start_servers() will register this function as an atexit
	handler for each server it starts, so there is normally no need to
	explicitly call this function.
	"""
	if verbose:
		print >>sys.stderr, "stopping http server on %s ..." % repr(httpd.server_address),
	httpd.shutdown()
	httpd_thread.join()
	if verbose:
		print >>sys.stderr, "done"
