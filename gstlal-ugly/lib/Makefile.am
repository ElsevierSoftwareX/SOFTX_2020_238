pkginclude_HEADERS = gstlal_sngltrigger.h sngltriggerrowtype.h
pkgconfig_DATA = gstlal-ugly.pc
lib_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = gstlal_sngltrigger.h gstlal_sngltrigger.c sngltriggerrowtype.h sngltriggerrowtype.c
libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
