AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalburst.la

lib@GSTPLUGINPREFIX@gstlalburst_la_SOURCES = \
	gstlalburst.c \
	gstlal_trigger.h gstlal_trigger.c \
	gstlal_string_triggergen.c gstlal_string_triggergen.h

lib@GSTPLUGINPREFIX@gstlalburst_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
lib@GSTPLUGINPREFIX@gstlalburst_la_LIBADD = $(top_builddir)/lib/gstlal-burst/libgstlalburst.la
lib@GSTPLUGINPREFIX@gstlalburst_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

