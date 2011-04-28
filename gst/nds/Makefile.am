INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstnds.la

libgstnds_la_SOURCES = ndssrc.h ndssrc.c
libgstnds_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(gstreamer_CFLAGS) $(NDS_CFLAGS)
libgstnds_la_LIBADD = $(top_srcdir)/lib/libgstlaltags.la
libgstnds_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(NDS_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

noinst_HEADERS = ndssrc.h
