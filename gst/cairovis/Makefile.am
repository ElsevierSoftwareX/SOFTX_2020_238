plugin_LTLIBRARIES = libcairovis.la

libcairovis_la_SOURCES = \
	cairovis.c \
	cairovis_base.h cairovis_base.c \
	cairovis_lineseries.h cairovis_lineseries.c \
	cairovis_waterfall.h cairovis_waterfall.c
libcairovis_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_video_CFLAGS) $(cairo_CFLAGS)
libcairovis_la_LIBADD = $(gstreamer_LIBS) $(gstreamer_video_LIBS) $(cairo_LIBS)
libcairovis_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)
