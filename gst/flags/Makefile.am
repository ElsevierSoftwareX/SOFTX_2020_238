plugin_LTLIBRARIES = libgstflags.la

libgstflags_la_SOURCES = gstflags.h gstflags.c 
libgstflags_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(gstreamer_CFLAGS) 
libgstflags_la_LIBADD = $(gstreamer_LIBS)
libgstflags_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)

noinst_HEADERS = gstflags.h
