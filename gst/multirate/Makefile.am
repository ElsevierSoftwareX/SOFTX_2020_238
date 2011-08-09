plugin_LTLIBRARIES = libgstmultirate.la

libgstmultirate_la_SOURCES = multiratefirinterp.h multiratefirinterp.c
libgstmultirate_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) \
	-DGST_LICENSE=\"LGPL\" -DGST_PACKAGE_NAME=PACKAGE_NAME -DGST_PACKAGE_ORIGIN=\"http://www.lsc-group.phys.uwm.edu/daswg\"
libgstmultirate_la_LIBADD = $(gstreamer_LIBS)
libgstmultirate_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)

noinst_HEADERS = multiratefirinterp.h
