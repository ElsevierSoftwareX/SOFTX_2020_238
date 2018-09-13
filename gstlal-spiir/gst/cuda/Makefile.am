ADD_CFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/lib/include
ADD_LIBS = -L\$(top_srcdir)/lib -lgstlalspiir -llapacke -llapack -lrefblas -lgfortran

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	cuda_plugin.c \
	spiir/spiir_kernel.cu \
	spiir/spiir.c \
	multiratespiir/multiratespiir_kernel.cu \
	multiratespiir/multiratespiir_utils.c \
	multiratespiir/multiratespiir.c \
	postcoh/postcoh_kernel.cu \
	postcoh/postcoh_utils.c \
	postcoh/postcoh.c \
	postcoh/postcohtable_utils.c \
	postcoh/postcoh_filesink.c \
	cohfar/knn_kde.c \
	cohfar/ssvkernel.c \
	cohfar/background_stats_utils.c \
	cohfar/cohfar_accumbackground.c \
	cohfar/cohfar_assignfar.c
#	deprecated
#	audioresample folder not working, seg fault
#	audioresample/cuda_gstaudioresample.c \
#	audioresample/speex_resampler_int.c \
#	audioresample/speex_resampler_float.c \
#	audioresample/speex_resampler_double.c
#	multidownsample folder not working, seg fault
#	multidownsample/gstlal_multidownsample.c

libcuda_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS) $(CHEALPIX_CFLAGS) $(ADD_CFLAGS)

libcuda_plugin_la_LIBADD = $(ADD_LIBS)

libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS) $(CHEALPIX_LIBS) -lstdc++ 

.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_LAL_CFLAGS) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) $(ADD_CFLAGS) --ptxas-options=-v -O0  -maxrregcount=0 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 --compiler-options=\"$(libgstlalspiir_la_CFLAGS)\" -c $<

noinst_HEADERS = \
	spiir/spiir_kernel.h \
	spiir/spiir.h \
	multiratespiir/multiratespiir_kernel.h \
	multiratespiir/multiratespiir_utils.h \
	multiratespiir/multiratespiir.h \
	postcoh/postcoh_utils.h \
	postcoh/postcoh.h \
	postcoh/postcohtable_utils.h \
	cohfar/background_stats.h \
	cohfar/background_stats_utils.h \
	cohfar/cohfar_accumbackground.h \
	cohfar/cohfar_assignfar.h

gstlal_cohfar_calc_fap_SOURCES = \
	cohfar/ssvkernel.c \
	cohfar/knn_kde.c \
	cohfar/background_stats_utils.c \
	cohfar/cohfar_calc_fap.c

gstlal_cohfar_calc_fap_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(ADD_CFLAGS)
gstlal_cohfar_calc_fap_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(ADD_LIBS)

gstlal_postcoh_gen_detrsp_map_SOURCES = \
	postcoh/postcoh_detrsp_map.c

gstlal_postcoh_gen_detrsp_map_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS) $(CHEALPIX_CFLAGS) $(ADD_CFLAGS)

gstlal_postcoh_gen_detrsp_map_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS) $(CHEALPIX_LIBS) $(ADD_LIBS) 


bin_PROGRAMS = \
	gstlal_cohfar_calc_fap \
	gstlal_postcoh_gen_detrsp_map

