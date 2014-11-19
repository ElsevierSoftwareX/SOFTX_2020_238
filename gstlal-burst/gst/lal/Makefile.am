AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstlalburst.la

libgstlalburst_la_SOURCES = \
	gstlalburst.c \
	gstlal_burst_triggergen.h gstlal_burst_triggergen.c
libgstlalburst_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalburst_la_LIBADD = $(top_builddir)/lib/libgstlalburst.la
libgstlalburst_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

