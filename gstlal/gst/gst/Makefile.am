AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlal_adder.la

libgstlal_adder_la_SOURCES = \
	gstadder.h gstadder.c \
	gstadderorc.h gstadderorc-dist.h gstadderorc-dist.c
libgstlal_adder_la_CFLAGS = $(AM_CFLAGS) $(ORC_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
libgstlal_adder_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlal.la $(top_builddir)/lib/gstlal/libgstlaltypes.la
libgstlal_adder_la_LDFLAGS = $(AM_LDFLAGS) $(ORC_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
