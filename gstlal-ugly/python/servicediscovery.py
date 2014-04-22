# Copyright (C) 2012--2014  Kipp Cannon
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


import avahi
import dbus
import socket


__all__ = ["DEFAULT_PROTO", "DEFAULT_DOMAIN", "ServiceInfo", "Publisher", "Listener", "ServiceBrowser"]


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                             Service Description
#
# =============================================================================
#


DEFAULT_PROTO = "_http._tcp."
DEFAULT_DOMAIN = "local."


#
# =============================================================================
#
#                              Service Publishing
#
# =============================================================================
#


class Publisher(object):
	def __init__(self):
		bus = dbus.SystemBus()
		server = dbus.Interface(bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
		self.group = dbus.Interface(bus.get_object(avahi.DBUS_NAME, server.EntryGroupNew()), avahi.DBUS_INTERFACE_ENTRY_GROUP)

	def addservice(self, serviceinfo):
		self.group.AddService(
			avahi.IF_UNSPEC,	# interface
			avahi.PROTO_UNSPEC,	# protocol
			dbus.UInt32(0),		# flags
			serviceinfo.getName().rstrip(".local."),	# service name
			serviceinfo.getType().rstrip(".local."),	# service type
			"local",	# FIXME	# domain
			"", # FIXME socket.inet_ntoa(serviceinfo.getAddress()),	# host/address
			dbus.UInt16(serviceinfo.getPort()),	# port
			serviceinfo.getText()	# text/description
		)
		self.group.Commit()

	def unpublish(self):
		self.group.Reset()

	def __del__(self):
		self.unpublish()


#
# =============================================================================
#
#                              Service Discovery
#
# =============================================================================
#


class Listener(object):
	def addService(self, stype, name, server, address, port, properties):
		pass

	def removeService(self, stype, name, server, address, port, properties):
		pass


class ServiceBrowser(object):
	def __init__(self, listener, stype = DEFAULT_PROTO + DEFAULT_DOMAIN):
		self.listener = listener
		bus = dbus.SystemBus()
		server = dbus.Interface(bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
		browser = dbus.Interface(bus.get_object(avahi.DBUS_NAME, server.ServiceBrowserNew(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, stype, "local", dbus.UInt32(0))), avahi.DBUS_INTERFACE_SERVICE_BROWSER)
		browser.connect_to_signal("ItemNew", self.__itemnew_handler)

	def __itemnew_handler(self, interface, protocol, name, stype, domain, flags):
		if flags & avahi.LOOKUP_RESULT_LOCAL:
			# local service
			pass
		self.listener.addService(stype, name, None, interface, None, None)


#
# =============================================================================
#
#                                     Demo
#
# =============================================================================
#


if __name__ == "__main__":
	#
	# usage:
	#
	# python /path/to/servicediscovery.py [publish]
	#
	# if publish is given on the command line then a service is
	# published, otherwise a browser is started and discovered services
	# are printed
	#

	import sys

	if sys.argv[-1] == "publish":
		#
		# publish a service
		#

		publisher = Publisher()
		publisher.addservice(ServiceInfo(
			DEFAULT_PROTO + DEFAULT_DOMAIN,
			"%s.%s" % ("My Test Service", DEFAULT_PROTO + DEFAULT_DOMAIN),
			address = socket.inet_aton("127.0.0.1"),
			port = 3000,
			server = socket.gethostname(),
			properties = {
				"version": "0.10",
				"a": "test value",
				"b": "another value"
			}
		))
		raw_input("Service published.  Press return to unpublish and quit.\n")
		publisher.unpublish()
	else:
		#
		# browse for services
		#

		class MyListener(Listener):
			def print_msg(self, action, stype, name, server, address, port, properties):
				print >>sys.stderr, "Service \"%s\" %s" % (name, action)
				print >>sys.stderr, "\tType is \"%s\"" % stype
				print >>sys.stderr, "\tServer is %s" % server
				print >>sys.stderr, "\tAddress is %s" % (address and socket.inet_ntoa(address))
				print >>sys.stderr, "\tPort is %s" % port
				print >>sys.stderr, "\tProperties are %s" % properties
				print >>sys.stderr, "Browsing for services.  Press return quit."
			def addService(self, stype, name, server, address, port, properties):
				self.print_msg("added", stype, name, server, address, port, properties)
			def removeService(self, stype, name, server, address, port, properties):
				self.print_msg("removed", stype, name, server, address, port, properties)
		browser = ServiceBrowser(MyListener())
		raw_input("Browsing for services.  Press return quit.\n")
