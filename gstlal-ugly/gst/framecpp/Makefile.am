INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libframecpp_plugin.la

libframecpp_plugin_la_SOURCES = framecpp_plugin.c framecpp_channeldemux.cc framecpp_channeldemux.h
libframecpp_plugin_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_CXXFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_LIBADD = $(top_builddir)/lib/libgstlal.la $(top_builddir)/lib/libgstlaltags.la $(top_builddir)/lib/libgstlaltypes.la
libframecpp_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(framecpp_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
