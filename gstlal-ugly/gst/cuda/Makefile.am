AM_CPPFLAGS = -I$(top_srcdir)/lib 

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	LIGOLw_xmllib/LIGOLwReader.c \
	LIGOLw_xmllib/LIGOLwUtils.c \
	cuda_plugin.c \
	iirfilter/iirfilter_kernel.cu \
	iirfilter/cuda_gstlal_iirbank.c \
	multiratespiir/multiratespiir_kernel.cu \
	multiratespiir/spiir_state_utils.c \
	multiratespiir/multiratespiir_utils.c \
	multiratespiir/multiratespiir.c \
	postcoh/postcoh.c

#	postcoh/detresponse_skymap.c
#	audioresample folder not working, seg fault
#	audioresample/cuda_gstaudioresample.c \
#	audioresample/speex_resampler_int.c \
#	audioresample/speex_resampler_float.c \
#	audioresample/speex_resampler_double.c

#	multidownsample folder not working, seg fault
#	multidownsample/gstlal_multidownsample.c

libcuda_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS) $(CHEALPIX_CFLAGS)

libcuda_plugin_la_LIBADD = 

libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS) $(CHEALPIX_LIBS) -llapack -llapacke

.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) --ptxas-options=-v -O0  -maxrregcount=0 -arch compute_30 -code compute_30,sm_30 --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<

noinst_HEADERS = \
	LIGOLw_xmllib/LIGOLwHeader.h \
	iirfilter/iirfilter_kernel.h \
	iirfilter/cuda_gstlal_iirbank.h \
	multiratespiir/multiratespiir.h \
	multiratespiir/spiir_state_macro.h \
	postcoh/postcoh.h

