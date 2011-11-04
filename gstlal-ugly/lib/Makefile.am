SUBDIRS = skymap

pkginclude_HEADERS = gstlal_snglinspiral.h gstlal_snglburst.h
pkgconfig_DATA = gstlalugly.pc
lib_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = gstlal_snglinspiral.h gstlal_snglinspiral.c gstlal_snglburst.h gstlal_snglburst.c
libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
