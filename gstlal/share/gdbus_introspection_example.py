import avahi
import sys
from gi.repository import Gio
from gi.repository import GLib

mainloop = GLib.MainLoop()

bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
server = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER, avahi.DBUS_INTERFACE_SERVER, None)

group_path = server.EntryGroupNew("()")
group = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, group_path, avahi.DBUS_INTERFACE_ENTRY_GROUP, None)

group_introspection = Gio.DBusProxy.new_sync(bus, Gio.DBusProxyFlags.NONE, None, avahi.DBUS_NAME, group_path, "org.freedesktop.DBus.Introspectable", None)
print(group_introspection.Introspect())
