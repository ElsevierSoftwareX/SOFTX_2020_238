AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

# FIXME: remove gstfdmemory.{c,h} when we can rely on gstreamer 1.6
EXTRA_DIST = \
	gstfdmemory.h gstfdmemory.c

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlal.la

lib@GSTPLUGINPREFIX@gstlal_la_SOURCES = \
	gstlal.c \
	gstlal_audioundersample.h gstlal_audioundersample.c \
	gstlal_autochisq.h gstlal_autochisq.c \
	gstlal_cachesrc.h gstlal_cachesrc.c \
	gstlal_drop.h gstlal_drop.c \
	gstlal_firbank.h gstlal_firbank.c \
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
lib@GSTPLUGINPREFIX@gstlal_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
lib@GSTPLUGINPREFIX@gstlal_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlal.la $(top_builddir)/lib/gstlal/libgstlaltags.la $(top_builddir)/lib/gstlal/libgstlaltypes.la
lib@GSTPLUGINPREFIX@gstlal_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
