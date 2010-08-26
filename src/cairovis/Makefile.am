plugin_LTLIBRARIES = libcairovis.la

libcairovis_la_SOURCES = \
	cairovis.c \
	cairovis_lineseries.h cairovis_lineseries.c
libcairovis_la_CFLAGS = -O2 $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${cairo_CFLAGS}
libcairovis_la_LIBADD = ${gstreamer_LIBS} ${cairo_LIBS}
libcairovis_la_LDFLAGS = ${GSTREAMER_PLUGIN_LDFLAGS}
