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
import sys


__all__ = ["DEFAULT_PROTO", "DEFAULT_DOMAIN", "Publisher", "Listener", "ServiceBrowser"]


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                            HTTP Service Metadata
#
# =============================================================================
#


DEFAULT_PROTO = "_http._tcp"
DEFAULT_DOMAIN = "local"


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

	def add_service(self, sname, stype, sdomain, host, port, properties = None):
		if properties is not None:
			assert not any("=" in key for key in properties)
		self.group.AddService(
			avahi.IF_UNSPEC,	# interface
			avahi.PROTO_INET,	# protocol
			dbus.UInt32(0),		# flags
			sname,			# service name
			stype,			# service type
			sdomain,		# domain
			host,			# host name
			dbus.UInt16(port),	# port
			avahi.dict_to_txt_array(properties if properties is not None else {})	# text/description
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
	def add_service(self, sname, stype, sdomain, host, port, properties):
		pass

	def remove_service(self, sname, stype, sdomain):
		pass

	def all_for_now(self):
		pass

	def failure(self, exception):
		pass


class ServiceBrowser(object):
	def __init__(self, mainloop, listener, stype = DEFAULT_PROTO, sdomain = DEFAULT_DOMAIN, ignore_local = False):
		self.listener = listener
		self.ignore_local = ignore_local
		bus = dbus.SystemBus(mainloop = mainloop)
		self.server = dbus.Interface(bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
		browser = dbus.Interface(bus.get_object(avahi.DBUS_NAME, self.server.ServiceBrowserNew(
			avahi.IF_UNSPEC,	# interface
			avahi.PROTO_UNSPEC,	# protocol
			stype,			# service type
			sdomain,		# service domain
			dbus.UInt32(0)		# flags
		)), avahi.DBUS_INTERFACE_SERVICE_BROWSER)
		browser.connect_to_signal("ItemNew", self.itemnew_handler)
		browser.connect_to_signal("ItemRemove", self.itemremove_handler)
		browser.connect_to_signal("AllForNow", self.allfornow_handler)

	def itemnew_handler(self, interface, protocol, sname, stype, sdomain, flags):
		if self.ignore_local and (flags & avahi.LOOKUP_RESULT_LOCAL):
			# local service (on this machine)
			return
		self.server.ResolveService(interface, protocol, sname, stype, sdomain, avahi.PROTO_UNSPEC, dbus.UInt32(0), reply_handler = self.itemnew_resolved_handler, error_handler = self.resolve_error)

	def itemnew_resolved_handler(self, interface, protocol, sname, stype, sdomain, host, aprotocol, address, port, txt, flags):
		self.listener.add_service(sname, stype, sdomain, host, port, dict(s.split("=", 1) for s in avahi.txt_array_to_string_array(txt)))

	def resolve_error(self, *args, **kwargs):
		print >>sys.stderr, "resolve error:", args, kwargs

	def itemremove_handler(self, interface, protocol, sname, stype, sdomain, flags):
		if self.ignore_local and (flags & avahi.LOOKUP_RESULT_LOCAL):
			# local service (on this machine)
			return
		self.listener.remove_service(sname, stype, sdomain)

	def allfornow_handler(self):
		self.listener.all_for_now()

	def failure_handler(self, exception):
		self.listener.failure(exception)


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

	from dbus.mainloop.glib import DBusGMainLoop
	import gobject
	import socket

	if sys.argv[-1] == "publish":
		#
		# publish a service
		#

		publisher = Publisher()
		publisher.add_service(
			sname = "My Test Service",
			stype = DEFAULT_PROTO,
			sdomain = DEFAULT_DOMAIN,
			host = socket.getfqdn(),
			port = 3456,
			properties = {
				"version": "0.10",
				"a": "test value",
				"b": "another value"
			}
		)
		raw_input("Service published.  Press return to unpublish and quit.\n")
		publisher.unpublish()
	else:
		#
		# browse for services
		#

		class MyListener(Listener):
			def print_msg(self, action, sname, stype, sdomain, host, port, properties):
				print >>sys.stderr, "Service \"%s\" %s" % (sname, action)
				print >>sys.stderr, "\tType is \"%s\"" % stype
				print >>sys.stderr, "\tDomain is \"%s\"" % sdomain
				print >>sys.stderr, "\tHost is \"%s\"" % host
				print >>sys.stderr, "\tPort is %s" % port
				print >>sys.stderr, "\tProperties are %s\n" % properties
			def add_service(self, sname, stype, sdomain, host, port, properties):
				self.print_msg("added", sname, stype, sdomain, host, port, properties)
			def remove_service(self, sname, stype, sdomain):
				self.print_msg("removed", sname, stype, sdomain, None, None, None)
		browser = ServiceBrowser(DBusGMainLoop(), MyListener())
		print "Browsing for services.  Press CTRL-C to quit.\n"
		gobject.MainLoop().run()
