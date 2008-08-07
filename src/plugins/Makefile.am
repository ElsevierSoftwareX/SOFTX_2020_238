plugin_LTLIBRARIES = libgstlal.la

libgstlal_la_SOURCES = \
	gstlal.h gstlal.c \
	gstlal_framesrc.h gstlal_framesrc.c \
	low_latency_inspiral_functions.h low_latency_inspiral_functions.c \
	gstlal_templatebank.h gstlal_templatebank.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS}
libgstlal_la_LIBADD = ${gstreamer_LIBS} ${LAL_LIBS} ${GSL_LIBS}
libgstlal_la_LDFLAGS = ${GSTREAMER_PLUGIN_LDFLAGS}

noinst_HEADERS = gstlal.h
