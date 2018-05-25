plugin_LTLIBRARIES = libgstcmath.la

# sources used to compile this plug-in
libgstcmath_la_SOURCES = \
	cmath.c \
	cmath_base.h cmath_base.c \
	cmath_cabs.c \
	cmath_creal.c \
	cmath_cimag.c \
	cmath_cexp.c \
	cmath_cln.c \
	cmath_clog.c \
	cmath_clog10.c \
	cmath_cpow.c
libgstcmath_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
libgstcmath_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
libgstcmath_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

# headers we need but don't want installed
noinst_HEADERS = cmath_base.h
