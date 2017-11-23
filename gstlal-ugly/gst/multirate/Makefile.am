noinst_HEADERS = multiratefirdecim.h multiratefirinterp.h

#plugin_LTLIBRARIES = libgstmultirate.la

libgstmultirate_la_SOURCES = multirate.c \
	multiratefirdecim.h multiratefirdecim.c \
	multiratefirinterp.h multiratefirinterp.c
libgstmultirate_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_LICENSE=\"LGPL\" -DGST_PACKAGE_NAME=\"$(PACKAGE_NAME)\" -DGST_PACKAGE_ORIGIN=\"http://www.lsc-group.phys.uwm.edu/daswg\"
libgstmultirate_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
