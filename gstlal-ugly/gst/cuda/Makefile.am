AM_CPPFLAGS = -I$(top_srcdir)/lib 

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	LIGOLw_xmllib/LIGOLwReader.c \
	LIGOLw_xmllib/LIGOLwWriter.c \
	LIGOLw_xmllib/LIGOLwUtils.c \
	cuda_plugin.c \
	spiir/spiir_kernel.cu \
	spiir/spiir.c \
	multiratespiir/multiratespiir_kernel.cu \
	multiratespiir/multiratespiir_utils.c \
	multiratespiir/multiratespiir.c \
	postcoh/postcoh_kernel.cu \
	postcoh/postcoh_utils.c \
	postcoh/postcoh.c \
	postcoh/postcohinspiral_table_utils.c \
	postcoh/postcoh_filesink.c \
	cohfar/ssvkernel.c \
	cohfar/background_stats_utils.c \
	cohfar/cohfar_accumbackground.c \
	cohfar/cohfar_assignfap.c

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

libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS) $(CHEALPIX_LIBS) -lstdc++ -llapacke -llapack -lrefblas -lgfortran

.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_LAL_CFLAGS) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) --ptxas-options=-v -O0  -maxrregcount=0 -arch compute_30 -code compute_30,sm_30 --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<

noinst_HEADERS = \
	LIGOLw_xmllib/LIGOLwHeader.h \
	spiir/spiir_kernel.h \
	spiir/spiir.h \
	multiratespiir/multiratespiir_kernel.h \
	multiratespiir/multiratespiir_utils.h \
	multiratespiir/multiratespiir.h \
	postcoh/postcoh_utils.h \
	postcoh/postcoh.h \
	postcoh/postcohinspiral_table.h \
	postcoh/postcohinspiral_table_utils.h \
	cohfar/background_stats.h \
	cohfar/background_stats_utils.h \
	cohfar/cohfar_accumbackground.h \
	cohfar/cohfar_assignfap.h

gstlal_cohfar_calc_cdf_SOURCES = \
	LIGOLw_xmllib/LIGOLwReader.c \
	LIGOLw_xmllib/LIGOLwWriter.c \
	LIGOLw_xmllib/LIGOLwUtils.c \
	cohfar/ssvkernel.c \
	cohfar/background_stats_utils.c \
	cohfar/cohfar_calc_cdf.c

gstlal_cohfar_calc_cdf_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) 
gstlal_cohfar_calc_cdf_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

bin_PROGRAMS = \
	gstlal_cohfar_calc_cdf
