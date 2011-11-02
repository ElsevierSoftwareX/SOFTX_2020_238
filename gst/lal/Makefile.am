INCLUDES = -I$(top_srcdir)/lib -I$(top_srcdir)/lib/skymap

plugin_LTLIBRARIES = libgstlal.la libgstlaladder.la

libgstlal_la_SOURCES = \
	gstlal_plugins.h gstlal.c \
	gstlal_framesrc.h gstlal_framesrc.c \
	gstlal_framesink.h gstlal_framesink.c \
	low_latency_inspiral_functions.h low_latency_inspiral_functions.c \
	gstlal_matrixmixer.h gstlal_matrixmixer.c \
	gstlal_simulation.h gstlal_simulation.c \
	gstlal_whiten.h gstlal_whiten.c \
	gstlal_nxydump.h gstlal_nxydump.c \
	gstlal_multiplier.h gstlal_multiplier.c \
	gstlal_coinc.h gstlal_coinc.c \
	gstlal_skymap.h gstlal_skymap.c \
	gstlal_triggergen.h gstlal_triggergen.c \
	gstlal_triggerxmlwriter.h gstlal_triggerxmlwriter.c \
	gstlal_gate.h gstlal_gate.c \
	gstlal_chisquare.h gstlal_chisquare.c \
	gstlal_autochisq.h gstlal_autochisq.c \
	gstlal_firbank.h gstlal_firbank.c \
	gstlal_sumsquares.h gstlal_sumsquares.c \
	gstlal_togglecomplex.h gstlal_togglecomplex.c \
	gstlal_nofakedisconts.h gstlal_nofakedisconts.c \
	gstlal_reblock.h gstlal_reblock.c \
	gstlal_delay.h gstlal_delay.c \
	gstlal_iirbank.h gstlal_iirbank.c \
	gstlal_mean.h gstlal_mean.c \
	gstlal_timeslicechisq.h gstlal_timeslicechisq.c \
	gstlal_specgram.h gstlal_specgram.c \
	gstlal_segmentsrc.h gstlal_segmentsrc.c \
	gstlal_blcbc_triggergen.h gstlal_blcbc_triggergen.c \
	gstlal_audioundersample.h gstlal_audioundersample.c \
	gstlal_drop.h gstlal_drop.c \
	gstlal_peak.h gstlal_peak.c \
	gstlal_itac.h gstlal_itac.c \
	gstlal_pad.h gstlal_pad.c \
	gstlal_trim.h gstlal_trim.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LIBADD = $(top_builddir)/lib/skymap/libwanalysis.la $(top_builddir)/lib/libgstlal.la $(top_builddir)/lib/libgstlaltags.la $(top_builddir)/lib/libgstlaltypes.la
libgstlal_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

libgstlaladder_la_SOURCES = \
	gstadder.h gstadder.c \
	gstadderorc.h gstadderorc-dist.h gstadderorc-dist.c
libgstlaladder_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(ORC_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
libgstlaladder_la_LIBADD = $(top_builddir)/lib/libgstlaltypes.la
libgstlaladder_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(ORC_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

noinst_HEADERS = gstlal_plugins.h
