AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libcontrol_plugin.la

libcontrol_plugin_la_SOURCES = \
	control_plugin.c \
	control_timeshift.h control_timeshift.c 
libcontrol_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gcontroler_CFLAGS)
libcontrol_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gcontroler_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
