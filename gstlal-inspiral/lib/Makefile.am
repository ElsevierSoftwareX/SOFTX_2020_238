SUBDIRS = skymap

pkginclude_HEADERS = gstlal_snglinspiral.h
pkgconfig_DATA = gstlal-inspiral.pc
lib_LTLIBRARIES = libgstlalinspiral.la

libgstlalinspiral_la_SOURCES = gstlal_snglinspiral.h gstlal_snglinspiral.c
libgstlalinspiral_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalinspiral_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
