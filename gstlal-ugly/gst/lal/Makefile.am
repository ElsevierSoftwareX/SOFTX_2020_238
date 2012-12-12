INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = \
	gstlalugly.c \
	gstlal_iirbank.h gstlal_iirbank.c \
	gstlal_mean.h gstlal_mean.c \
	gstlal_multiplier.h gstlal_multiplier.c \
	gstlal_pad.h gstlal_pad.c \
	gstlal_specgram.h gstlal_specgram.c \
	gstlal_trim.h gstlal_trim.c \
	gstlal_burst_triggergen.h gstlal_burst_triggergen.c \
	gstlal_bitvectorgen.h gstlal_bitvectorgen.c \
	gstlal_cachesrc.h gstlal_cachesrc.c
libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalugly_la_LIBADD = $(top_builddir)/lib/libgstlalugly.la
libgstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
