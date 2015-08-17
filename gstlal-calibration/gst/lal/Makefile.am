AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlalcalibration.la libgstlalinterleave.la

libgstlalcalibration_la_SOURCES = \
	gstlalcalibration.c \
        gstlal_wings.c gstlal_wings.h 
libgstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
#libgstlalcalibration_la_LIBADD = $(top_builddir)/lib/libgstlalcalibration.la
libgstlalcalibration_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

libgstlalinterleave_la_SOURCES = \
	gstinterleave.c gstinerleave.h
libgstlalinterleave_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
#libgstlalinterleave_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlaltypes.la
libgstlalinterleave_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
