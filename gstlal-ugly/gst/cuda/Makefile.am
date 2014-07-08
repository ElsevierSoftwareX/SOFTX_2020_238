AM_CPPFLAGS = -I$(top_srcdir)/lib 

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	cuda_plugin.c \
	iirfilter/iirfilter_kernel.cu \
	iirfilter/cuda_gstlal_iirbank.c 

#	audioresample folder not working, seg fault
#	audioresample/cuda_gstaudioresample.c \
#	audioresample/speex_resampler_int.c \
#	audioresample/speex_resampler_float.c \
#	audioresample/speex_resampler_double.c

#	multidownsample folder not working, seg fault
#	multidownsample/gstlal_multidownsample.c

libcuda_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS)

libcuda_plugin_la_LIBADD = 

libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS)

.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) --ptxas-options=-v -O3  -maxrregcount=0 -arch compute_30 -code compute_30,sm_30 --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<

noinst_HEADERS = \
	iirfilter/iirfilter_kernel.h \
	iirfilter/cuda_gstlal_iirbank.h \
	multiratespiir/multiratespiir.h \
	multiratespiir/spiir_state_macro.h \
	multiratespiir/resampler_state_macro.h

