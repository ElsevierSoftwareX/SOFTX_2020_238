AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstgds.la

libgstgds_la_SOURCES = \
	gds_plugin.c \
	framexmitsink.h framexmitsink.cc \
	framexmitsrc.h framexmitsrc.cc \
	lvshmsrc.h lvshmsrc.cc \
	lvshmsink.h lvshmsink.cc
libgstgds_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gds_CFLAGS)
libgstgds_la_CXXFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gds_CFLAGS) $(gds_framexmit_CFLAGS)
libgstgds_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gds_LIBS) $(gds_framexmit_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
