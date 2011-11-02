INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libframecpp_plugin.la

libframecpp_plugin_la_SOURCES = framecpp_plugin.c framecpp_channeldemux.cc framecpp_channeldemux.h
libframecpp_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_CXXFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LDFLAGS) $(gstreamer_LIBS) $(framecpp_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
