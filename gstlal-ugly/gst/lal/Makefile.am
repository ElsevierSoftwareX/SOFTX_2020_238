AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = \
	gstlalugly.c \
	audioratefaker.h audioratefaker.c \
	gstlal_iirbank.h gstlal_iirbank.c \
	gstlal_interpolator.h gstlal_interpolator.c \
	gstlal_mean.h gstlal_mean.c \
	gstlal_multiplier.h gstlal_multiplier.c \
	gstlal_pad.h gstlal_pad.c \
	gstlal_specgram.h gstlal_specgram.c \
	gstlal_trim.h gstlal_trim.c \
	gstlal_bitvectorgen.h gstlal_bitvectorgen.c
libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
