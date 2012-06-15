# Copyright (C) 2012  Kipp Cannon
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


import socket


from gstlal.pyzeroconf import Zeroconf


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


DEFAULT_STYPE = "_http._tcp.local."


from gstlal.pyzeroconf.Zeroconf import ServiceInfo


#
# =============================================================================
#
#                              Service Publishing
#
# =============================================================================
#


try:
	#
	# avahi is the preferred way to do this.  as a system-level service
	# it allows multiple applications to advertise services without
	# contention (there can be only one responder per host), and the
	# daemon can cache service announcements even when the application
	# that cares about them has not yet started
	#

	import avahi
	import dbus

	_zc = None

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

except ImportError:
	#
	# the Zeroconf module provides a pure python responder
	# implementation that we can fall back on if avahi isn't available
	#

	_zc = Zeroconf.Zeroconf(bindaddress = "0.0.0.0")

	class Publisher(object):
		def __init__(self):
			self.group = set()

		def addservice(self, serviceinfo):
			_zc.registerService(serviceinfo)
			self.group.add(serviceinfo)

		def unpublish(self):
			while self.group:
				_zc.unregisterService(self.group.pop())

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
	def addService(self, stype, name, address, port, properties):
		pass

	def removeService(self, stype, name, address, port, properties):
		pass


if _zc is None:

	class ServiceBrowser(object):
		def __init__(self, listener, stype = DEFAULT_STYPE):
			self.listener = listener
			bus = dbus.SystemBus()
			server = dbus.Interface(bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
			browser = dbus.Interface(bus.get_object(avahi.DBUS_NAME, server.ServiceBrowserNew(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, stype, "local", dbus.UInt32(0))), avahi.DBUS_INTERFACE_SERVICE_BROWSER)
			browser.connect_to_signal("ItemNew", self.__itemnew_handler)

		def __itemnew_handler(self, interface, protocol, name, stype, domain, flags):
			if flags & avahi.LOOKUP_RESULT_LOCAL:
				# local service
				pass
			self.listener.addService(stype, name, interface, None, None)

else:

	class ServiceBrowser(Zeroconf.ServiceBrowser):
		def __init__(self, listener, stype = DEFAULT_STYPE):
			self._listener = listener
			super(type(self), self).__init__(_zc, stype, self)

		def addService(self, zc, stype, name):
			info = zc.getServiceInfo(stype, name)
			if info is not None:
				self._listener.addService(stype, name, info.getAddress(), info.getPort(), info.getProperties())
			else:
				self._listener.addService(stype, name, None, None, None)

		def removeService(self, zc, stype, name):
			info = zc.getServiceInfo(stype, name)
			if info is not None:
				self._listener.removeService(stype, name, info.getAddress(), info.getPort(), info.getProperties())
			else:
				self._listener.removeService(stype, name, None, None, None)


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
			DEFAULT_STYPE,
			"%s.%s" % ("My Test Service", DEFAULT_STYPE),
			address = socket.inet_aton("127.0.0.1"),
			port = 3000,
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
			def addService(self, stype, name, address, port, properties):
				print >>sys.stderr, "Service \"%s\" added" % name
				print >>sys.stderr, "\tType is \"%s\"" % stype
				print >>sys.stderr, "\tAddress is %s" % (address and socket.inet_ntoa(address))
				print >>sys.stderr, "\tPort is %s" % port
				print >>sys.stderr, "\tProperties are %s" % properties
				print >>sys.stderr, "Browsing for services.  Press return quit."
		browser = ServiceBrowser(MyListener())
		raw_input("Browsing for services.  Press return quit.\n")
