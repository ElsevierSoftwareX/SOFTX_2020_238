plugin_LTLIBRARIES = libcairovis.la

# FIXME: is it OK to have Matplotlib as a build dependency?
#colormap_data.c colormap_data.h: gencolormap.py
#	python $<

libcairovis_la_SOURCES = \
	cairovis.c \
	colormap_data.h colormap_data.c \
	colormap.h colormap.c \
	cairovis_base.h cairovis_base.c \
	cairovis_histogram.h cairovis_histogram.c \
	cairovis_lineseries.h cairovis_lineseries.c \
	cairovis_waterfall.h cairovis_waterfall.c
libcairovis_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_video_CFLAGS) $(cairo_CFLAGS) $(GSL_CFLAGS)
libcairovis_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(gstreamer_video_LIBS) $(cairo_LIBS) $(GSL_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
