AM_CPPFLAGS = -I$(top_srcdir)/lib 

plugin_LTLIBRARIES = libcuda_plugin.la

libcuda_plugin_la_SOURCES = \
	cuda_plugin.c \
	cuda_iirbank.h cuda_iirbank.cu \
	gstlal_iirbankCuda.h gstlal_iirbankCuda.c 
libcuda_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(AM_CPPFLAGS) $(NVCC_CFLAGS)
libcuda_plugin_la_LIBADD = 
libcuda_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(NVCC_LIBS)
.cu.lo:
	$(top_srcdir)/gnuscripts/cudalt.py $@ $(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(NVCC_GSTLAL_CFLAGS) $(NVCC_gstreamer_CFLAGS) --ptxas-options=-v -O3  -maxrregcount=0 -arch compute_30 -code compute_30,sm_30 --compiler-options=\"$(libgstlalugly_la_CFLAGS)\" -c $<
