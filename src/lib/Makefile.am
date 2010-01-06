pkginclude_HEADERS = gstlal.h

lib_LTLIBRARIES = libgstlal.la

libgstlal_la_SOURCES = gstlal.c gstlal.h
libgstlal_la_CFLAGS = $(GSL_CLFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LDFLAGS = $(GSL_LDFLAGS) $(LAL_LDFLAGS) $(gstreamer_LDFLAGS)
