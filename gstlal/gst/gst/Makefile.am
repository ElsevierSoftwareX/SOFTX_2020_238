AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlaladder.la

libgstlaladder_la_SOURCES = \
	gstadder.h gstadder.c \
	gstadderorc.h gstadderorc-dist.h gstadderorc-dist.c
libgstlaladder_la_CFLAGS = $(AM_CFLAGS) $(ORC_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
libgstlaladder_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlaltypes.la
libgstlaladder_la_LDFLAGS = $(AM_LDFLAGS) $(ORC_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
