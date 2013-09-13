AM_CPPFLAGS = -I$(top_srcdir)/lib/skymap -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstlalinspiral.la

libgstlalinspiral_la_SOURCES = \
	gstlalinspiral.c \
	gstlal_blcbc_triggergen.h gstlal_blcbc_triggergen.c \
	gstlal_coinc.h gstlal_coinc.c \
	gstlal_itac.h gstlal_itac.c \
	gstlal_skymap.h gstlal_skymap.c \
	gstlal_timeslicechisq.h gstlal_timeslicechisq.c \
	gstlal_triggergen.h gstlal_triggergen.c
libgstlalinspiral_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalinspiral_la_LIBADD = $(top_builddir)/lib/libgstlalinspiral.la  $(top_builddir)/lib/skymap/libwanalysis.la
libgstlalinspiral_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
