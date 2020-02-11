AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalugly.la

lib@GSTPLUGINPREFIX@gstlalugly_la_SOURCES = \
	gstlalugly.c \
	audioratefaker.h audioratefaker.c \
	gstlal_aggregator.h gstlal_aggregator.c \
	gstlal_bitvectorgen.h gstlal_bitvectorgen.c \
	gstlaldeglitchfilter.h gstlaldeglitchfilter.c \
	gstlal_iirbank.h gstlal_iirbank.c \
	gstlal_interpolator.h gstlal_interpolator.c \
	gstlal_tdwhiten.h gstlal_tdwhiten.c \
	gstlal_latency.h gstlal_latency.c \
	#gstlal_specgram.h gstlal_specgram.c \
	#gstlal_mean.h gstlal_mean.c \
	#gstlal_pad.h gstlal_pad.c \
	#gstlal_trim.h gstlal_trim.c

lib@GSTPLUGINPREFIX@gstlalugly_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
#lib@GSTPLUGINPREFIX@gstlalugly_la_LIBADD = $(top_builddir)/lib/libgstlalugly.la
lib@GSTPLUGINPREFIX@gstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS)  $(GSTLAL_PLUGIN_LDFLAGS)
