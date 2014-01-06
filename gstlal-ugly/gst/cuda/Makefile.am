AM_CPPFLAGS = -I$(top_srcdir)/lib 

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	cuda_plugin.c \
	iirfilter/cuda_iirbank.cu \
	iirfilter/gstlal_iirbankCuda.c \
	audioresample/gstaudioresample.c \
	audioresample/speex_resampler_int.c \
	audioresample/speex_resampler_float.c \
	audioresample/speex_resampler_double.c

libcuda_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS)

libcuda_plugin_la_LIBADD = 

libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS)

.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) --ptxas-options=-v -O3  -maxrregcount=0 -arch compute_30 -code compute_30,sm_30 --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<

noinst_HEADERS = \
	iirfilter/cuda_iirbank.h \
	iirfilter/gstlal_iirbankCuda.h \
	audioresample/arch.h \
	audioresample/fixed_arm4.h \
	audioresample/fixed_arm5e.h \
	audioresample/fixed_bfin.h \
	audioresample/fixed_debug.h \
	audioresample/fixed_generic.h \
	audioresample/gstaudioresample.h \
	audioresample/resample.c \
	audioresample/resample_sse.h \
	audioresample/speex_resampler.h \
	audioresample/speex_resampler_wrapper.h
 
