INCLUDES = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlal.la libgstlaladder.la

libgstlal_la_SOURCES = \
	gstlal.c \
	gstlal_audioundersample.h gstlal_audioundersample.c \
	gstlal_autochisq.h gstlal_autochisq.c \
	gstlal_drop.h gstlal_drop.c \
	gstlal_firbank.h gstlal_firbank.c \
	gstlal_framesink.h gstlal_framesink.c \
	gstlal_gate.h gstlal_gate.c \
	gstlal_matrixmixer.h gstlal_matrixmixer.c \
	gstlal_nofakedisconts.h gstlal_nofakedisconts.c \
	gstlal_nxydump.h gstlal_nxydump.c \
	gstlal_peak.h gstlal_peak.c \
	gstlal_reblock.h gstlal_reblock.c \
	gstlal_segmentsrc.h gstlal_segmentsrc.c \
	gstlal_simulation.h gstlal_simulation.c \
	gstlal_sumsquares.h gstlal_sumsquares.c \
	gstlal_statevector.h gstlal_statevector.c \
	gstlal_togglecomplex.h gstlal_togglecomplex.c \
	gstlal_whiten.h gstlal_whiten.c \
	gstlal_shift.h gstlal_shift.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LIBADD = $(top_builddir)/lib/libgstlal.la $(top_builddir)/lib/libgstlaltags.la $(top_builddir)/lib/libgstlaltypes.la
libgstlal_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

libgstlaladder_la_SOURCES = \
	gstadder.h gstadder.c \
	gstadderorc.h gstadderorc-dist.h gstadderorc-dist.c
libgstlaladder_la_CFLAGS = $(AM_CFLAGS) $(ORC_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
libgstlaladder_la_LIBADD = $(top_builddir)/lib/libgstlaltypes.la
libgstlaladder_la_LDFLAGS = $(AM_LDFLAGS) $(ORC_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
