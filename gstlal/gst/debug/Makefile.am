AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlaldebug.la

libgstlaldebug_la_SOURCES = \
	gstlaldebugplugin.c \
	tsvdec.h tsvdec.c
libgstlaldebug_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstlaldebug_la_LIBADD = 
libgstlaldebug_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
