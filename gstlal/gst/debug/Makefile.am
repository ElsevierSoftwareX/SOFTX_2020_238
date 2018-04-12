AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

#plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlaldebug.la

lib@GSTPLUGINPREFIX@gstlaldebug_la_SOURCES = \
	gstlaldebugplugin.c \
	tsvdec.h tsvdec.c
lib@GSTPLUGINPREFIX@gstlaldebug_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
lib@GSTPLUGINPREFIX@gstlaldebug_la_LIBADD = 
lib@GSTPLUGINPREFIX@gstlaldebug_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
