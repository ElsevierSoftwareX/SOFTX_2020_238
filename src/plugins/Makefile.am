plugin_LTLIBRARIES = libgstlal.la

libgstlal_la_SOURCES = gstlal.h gstlal.c gstlal_framesrc.h gstlal_framesrc.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS}
libgstlal_la_LIBADD = ${gstreamer_LIBS} ${LAL_LIBS}
libgstlal_la_LDFLAGS = ${GSTREAMER_PLUGIN_LDFLAGS}

noinst_HEADERS = gstlal.h
