AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = \
	gstlalugly.c \
	audioratefaker.h audioratefaker.c \
	gstlal_bitvectorgen.h gstlal_bitvectorgen.c \
	gstlal_interpolator.h gstlal_interpolator.c \
	gstlal_tdwhiten.h gstlal_tdwhiten.c \
	gstlal_trigger.h gstlal_trigger.c \
	gstlal_latency.h gstlal_latency.c \
	#gstlal_specgram.h gstlal_specgram.c \
	#gstlal_mean.h gstlal_mean.c \
	#gstlal_pad.h gstlal_pad.c \
	#gstlal_trim.h gstlal_trim.c

libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LIBADD = $(top_builddir)/lib/libgstlalugly.la
libgstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
