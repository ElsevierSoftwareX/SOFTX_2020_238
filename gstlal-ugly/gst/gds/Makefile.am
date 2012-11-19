INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgds_plugin.la

libgds_plugin_la_SOURCES = \
	gds_plugin.c \
	lvshmsrc.h lvshmsrc.c \
	lvshmsink.h lvshmsink.cc
libgds_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gds_CFLAGS)
libgds_plugin_la_CXXFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gds_CFLAGS)
libgds_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gds_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
