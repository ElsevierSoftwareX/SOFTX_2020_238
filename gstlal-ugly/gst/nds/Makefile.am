AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstnds.la

libgstnds_la_SOURCES = ndssrc.h ndssrc.c
libgstnds_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(NDS_CFLAGS)
libgstnds_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(NDS_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

noinst_HEADERS = ndssrc.h
