plugin_LTLIBRARIES = libgstaudiodelay.la

libgstaudiodelay_la_SOURCES = audiodelay.h audiodelay.c
libgstaudiodelay_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) \
	-DGST_LICENSE=\"LGPL\" -DGST_PACKAGE_NAME=PACKAGE_NAME -DGST_PACKAGE_ORIGIN=\"http://www.lsc-group.phys.uwm.edu/daswg\"
libgstaudiodelay_la_LIBADD = $(gstreamer_LIBS) $(gstreamer_audio_LIBS)
libgstaudiodelay_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)

noinst_HEADERS = audiodelay.h
