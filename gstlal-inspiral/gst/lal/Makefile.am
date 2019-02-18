AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_srcdir)/lib/skymap

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalinspiral.la

lib@GSTPLUGINPREFIX@gstlalinspiral_la_SOURCES = \
	gstlalinspiral.c \
	gstlal_itacac.h gstlal_itacac.c
#gstlal_triggergen.h gstlal_triggergen.c
#gstlal_blcbc_triggergen.h gstlal_blcbc_triggergen.c \
#gstlal_coinc.h gstlal_coinc.c \
#gstlal_skymap.h gstlal_skymap.c \
#gstlal_timeslicechisq.h gstlal_timeslicechisq.c
lib@GSTPLUGINPREFIX@gstlalinspiral_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
lib@GSTPLUGINPREFIX@gstlalinspiral_la_LIBADD = $(top_builddir)/lib/libgstlalinspiral.la  $(top_builddir)/lib/skymap/libwanalysis.la
lib@GSTPLUGINPREFIX@gstlalinspiral_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
