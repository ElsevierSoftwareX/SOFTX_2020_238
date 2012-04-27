INCLUDES = -I$(top_srcdir)/lib/skymap -I$(top_srcdir)/lib -I$(top_srcdir)/gst/lal

plugin_LTLIBRARIES = libgstlalugly.la

libgstlalugly_la_SOURCES = \
	gstlalugly.c \
	gstlal_blcbc_triggergen.h gstlal_blcbc_triggergen.c \
	gstlal_iirbank.h gstlal_iirbank.c\
	gstlal_coinc.h gstlal_coinc.c \
	gstlal_itac.h gstlal_itac.c \
	gstlal_mean.h gstlal_mean.c \
	gstlal_multiplier.h gstlal_multiplier.c \
	gstlal_pad.h gstlal_pad.c \
	gstlal_skymap.h gstlal_skymap.c \
	gstlal_specgram.h gstlal_specgram.c \
	gstlal_timeslicechisq.h gstlal_timeslicechisq.c \
	gstlal_triggergen.h gstlal_triggergen.c \
	gstlal_trim.h gstlal_trim.c \
	gstlal_burst_triggergen.h gstlal_burst_triggergen.c 

if COND_CUDA
libgstlalugly_la_SOURCES += cuda_iirbank.h cuda_iirbank.cu gstlal_iirbankCuda.h gstlal_iirbankCuda.c
endif

libgstlalugly_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(FFTW_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(INCLUDES) $(NVCC_CFLAGS)
libgstlalugly_la_LIBADD = $(top_builddir)/lib/libgstlalugly.la  $(top_builddir)/lib/skymap/libwanalysis.la
libgstlalugly_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(FFTW_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS)
.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<
