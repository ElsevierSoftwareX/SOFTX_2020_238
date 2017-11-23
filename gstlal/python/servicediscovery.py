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


from gi.repository import Gio


__all__ = ["DEFAULT_SERVICE_TYPE", "DEFAULT_DOMAIN", "Publisher", "Listener", "ServiceBrowser"]


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


DEFAULT_SERVICE_TYPE = "_http._tcp"
DEFAULT_DOMAIN = "local"


#
# =============================================================================
#
#                              Service Publishing
#
# =============================================================================
#


class Publisher(object):
	"""
	Glue code to connect to the avahi daemon through dbus and manage
	the advertisement of services.
	"""
	def __init__(self):
		bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
		server = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER, avahi.DBUS_INTERFACE_SERVER, None)
		group_path = server.EntryGroupNew("()")
		self.group = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, group_path, avahi.DBUS_INTERFACE_ENTRY_GROUP, None)

	def add_service(self, sname, port, stype = DEFAULT_SERVICE_TYPE, sdomain = DEFAULT_DOMAIN, host = "", properties = None):
		"""
		Add a service to the collection of services currently
		advertised.  sname and port specify the service name and
		the port number on which the service can be found.  stype
		and sdomain set the service type and service domain.

		Avahi is asked to advertise the service on all network
		interfaces to which it is connected.  If host is "" (the
		default) then on each interface avahi will use the host
		name corresponding to that network interface (as determined
		by itself).  This is a convenient way to ensure the service
		is advertised correctly on machines with multiple
		interfaces.

		If properties is not None it must be a dictionary of
		name-value pairs all of which are strings.  "=" is not
		allowed in any of the names.
		"""
		if properties is not None:
			assert not any("=" in key for key in properties)
		self.group.AddService(
			"(iiussssqaay)",
			avahi.IF_UNSPEC,	# interface
			avahi.PROTO_INET,	# protocol
			0,			# flags
			sname,			# service name
			stype,			# service type
			sdomain,		# domain
			host,			# host name
			port,			# port
			avahi.dict_to_txt_array(properties if properties is not None else {})	# text/description
		)
		self.group.Commit("()")

	def unpublish(self):
		"""
		Unpublish all services.
		"""
		self.group.Reset("()")

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
	"""
	Parent class for Listener implementations.  Each method corresponds
	to an event type.  Subclasses override the desired methods with the
	code to be invoked upon those events.  The default methods are all
	no-ops.  An instance of a Listener implementation is required to
	initialize a ServiceBrowser.
	"""
	def add_service(self, sname, stype, sdomain, host, port, properties):
		pass

	def remove_service(self, sname, stype, sdomain):
		pass

	def all_for_now(self):
		pass

	def failure(self, *args):
		pass


class ServiceBrowser(object):
	"""
	Glue code to connect a Listener implementation to the avahi daemon
	through dbus.
	"""
	def __init__(self, listener, stype = DEFAULT_SERVICE_TYPE, sdomain = DEFAULT_DOMAIN, ignore_local = False):
		"""
		Connects to the avahi daemon through dbus, requests an
		avahi ServiceBrowser instance from the daemon configured to
		browse for the given service type and domain, then connects
		signal handlers that forward information from avahi to the
		methods of a Listener instance.

		listener is an instance of a subclass of Listener (or any
		other object that provides the required methods to be used
		as call-backs).

		if ignore_local is True then services discovered on the
		local machine itself will be ignored (the default is False,
		all discovered services are reported to the Listener).
		"""
		self.listener = listener
		self.ignore_local = ignore_local
		bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
		self.server = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER, avahi.DBUS_INTERFACE_SERVER, None)
		browser_path = self.server.ServiceBrowserNew(
			"(iissu)",
			avahi.IF_UNSPEC,	# interface
			avahi.PROTO_UNSPEC,	# protocol
			stype,			# service type
			sdomain,		# service domain
			0			# flags
		)
		bus.signal_subscribe(None, None, "ItemNew", browser_path, None, Gio.DBusSignalFlags.NONE, self.itemnew_handler, None)
		bus.signal_subscribe(None, None, "ItemRemove", browser_path, None, Gio.DBusSignalFlags.NONE, self.itemremove_handler, None)
		bus.signal_subscribe(None, None, "AllForNow", browser_path, None, Gio.DBusSignalFlags.NONE, self.allfornow_handler, None)
		bus.signal_subscribe(None, None, "Failure", browser_path, None, Gio.DBusSignalFlags.NONE, self.failure_handler, None)

	def itemnew_handler(self, bus, sender_name, object_path, interface_name, signal_name, (interface, protocol, sname, stype, sdomain, flags), data):
		"""
		Internal ItemNew signal handler.  Forwards the essential
		information to the Listener's .add_service() method.
		"""
		if self.ignore_local and (flags & avahi.LOOKUP_RESULT_LOCAL):
			# local service (on this machine)
			return
		interface, protocol, sname, stype, sdomain, host, aprotocol, address, port, txt, flags = self.server.ResolveService(
			"(iisssiu)",
			interface,
			protocol,
			sname,
			stype,
			sdomain,
			avahi.PROTO_UNSPEC,
			0
		)
		self.listener.add_service(sname, stype, sdomain, host, port, dict(s.split("=", 1) for s in avahi.txt_array_to_string_array(txt)))

	def itemremove_handler(self, bus, sender_name, object_path, interface_name, signal_name, (interface, protocol, sname, stype, sdomain, flags), data):
		"""
		Internal ItemRemove signal handler.  Forwards the essential
		information to the Listener's .remove_service() method.
		"""
		if self.ignore_local and (flags & avahi.LOOKUP_RESULT_LOCAL):
			# local service (on this machine)
			return
		self.listener.remove_service(sname, stype, sdomain)

	def allfornow_handler(self, bus, sender_name, object_path, interface_name, signal_name, parameters, data):
		"""
		Internal AllForNow signal handler.  Forwards the essential
		information to the Listener's .all_for_now() method.
		"""
		self.listener.all_for_now()

	def failure_handler(self, bus, sender_name, object_path, interface_name, signal_name, parameters, data):
		"""
		Internal Failure signal handler.  Forwards the essential
		information to the Listener's .failure() method.
		"""
		self.listener.failure(*parameters)


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

	from gi.repository import GLib
	import sys

	if sys.argv[-1] == "publish":
		#
		# publish a service
		#

		publisher = Publisher()
		publisher.add_service(
			sname = "My Test Service",
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
			def failure(self, *args):
				print >>sys.stderr, "failure", args
		mainloop = GLib.MainLoop()
		browser = ServiceBrowser(MyListener())
		print "Browsing for services.  Press CTRL-C to quit.\n"
		mainloop.run()
