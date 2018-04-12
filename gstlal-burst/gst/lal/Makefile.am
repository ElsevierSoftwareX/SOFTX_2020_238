AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalburst.la

lib@GSTPLUGINPREFIX@gstlalburst_la_SOURCES = \
	gstlalburst.c
lib@GSTPLUGINPREFIX@gstlalburst_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
lib@GSTPLUGINPREFIX@gstlalburst_la_LIBADD = $(top_builddir)/lib/libgstlalburst.la
lib@GSTPLUGINPREFIX@gstlalburst_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

