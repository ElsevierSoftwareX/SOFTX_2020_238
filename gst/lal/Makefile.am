INCLUDES = -I$(top_srcdir)/lib -I$(top_srcdir)/lib/skymap

plugin_LTLIBRARIES = libgstlal.la

libgstlal_la_SOURCES = \
	gstlal_plugins.h gstlal.c \
	gstlal_framesrc.h gstlal_framesrc.c \
	gstlal_framesink.h gstlal_framesink.c \
	low_latency_inspiral_functions.h low_latency_inspiral_functions.c \
	gstlal_matrixmixer.h gstlal_matrixmixer.c \
	gstlal_simulation.h gstlal_simulation.c \
	gstlal_whiten.h gstlal_whiten.c \
	gstlal_nxydump.h gstlal_nxydump.c \
	gstlalcollectpads.h gstlalcollectpads.c \
	gstadder.h gstadder.c \
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
	gstlal_trim.h gstlal_trim.c \
	gstlal_pad.h gstlal_pad.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS)
libgstlal_la_LIBADD = $(gstreamer_LIBS) $(LAL_LIBS) $(GSL_LIBS) $(FFTW_LIBS) $(top_builddir)/lib/skymap/libwanalysis.la $(top_builddir)/lib/libgstlal.la $(top_builddir)/lib/libgstlaltags.la
libgstlal_la_LDFLAGS = $(AM_LDFLAGS) $(GSTREAMER_PLUGIN_LDFLAGS)

noinst_HEADERS = gstlal_plugins.h
