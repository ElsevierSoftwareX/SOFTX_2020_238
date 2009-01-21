plugin_LTLIBRARIES = libgstlalspeexresample.la

libgstlalspeexresample_la_SOURCES = \
        gstspeexresample.c \
        speex_resampler_int.c \
        speex_resampler_float.c \
        speex_resampler_double.c
libgstlalspeexresample_la_CFLAGS = -O2 $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS} ${liboil_CFLAGS}
libgstlalspeexresample_la_LIBADD = ${gstreamer_LIBS} ${LAL_LIBS} ${GSL_LIBS} ${liboil_LIBS}
libgstlalspeexresample_la_LDFLAGS = ${GSTREAMER_PLUGIN_LDFLAGS}

noinst_HEADERS = \
	arch.h \
	fixed_arm4.h \
	fixed_arm5e.h \
	fixed_bfin.h \
	fixed_debug.h \
	fixed_generic.h \
	gstspeexresample.h \
	resample.c \
	resample_sse.h \
	speex_resampler.h \
	speex_resampler_wrapper.h
