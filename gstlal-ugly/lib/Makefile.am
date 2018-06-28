pkginclude_HEADERS =
pkgconfig_DATA = gstlal-ugly.pc
lib_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES =
libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
