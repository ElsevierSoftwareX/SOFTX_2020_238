plugin_LTLIBRARIES = libgstprogressreport2.la

libgstprogressreport2_la_SOURCES = progressreport.h progressreport.c
libgstprogressreport2_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(gstreamer_CFLAGS) \
	-DGST_LICENSE=\"LGPL\" -DGST_PACKAGE_NAME=PACKAGE_NAME -DGST_PACKAGE_ORIGIN=\"http://www.lsc-group.phys.uwm.edu/daswg\"
libgstprogressreport2_la_LIBADD = $(gstreamer_LIBS)
libgstprogressreport2_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)

noinst_HEADERS = progressreport.h
