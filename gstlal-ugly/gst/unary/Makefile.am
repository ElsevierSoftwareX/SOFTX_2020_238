plugin_LTLIBRARIES = libgstunary.la

libgstunary_la_SOURCES = \
	unary.c \
	unary_base.h unary_base.c \
	unary_abs.c \
	unary_exp.c \
	unary_ln.c \
	unary_log.c \
	unary_log10.c \
	unary_pow.c
libgstunary_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
libgstunary_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
