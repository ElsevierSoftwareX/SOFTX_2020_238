AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@cmath.la

# sources used to compile this plug-in
lib@GSTPLUGINPREFIX@cmath_la_SOURCES = \
	cmath.c \
	cmath_base.h cmath_base.c \
	cmath_cabs.c \
	cmath_creal.c \
	cmath_cimag.c \
	cmath_cexp.c \
	cmath_carg.c \
	cmath_cln.c \
	cmath_clog.c \
	cmath_clog10.c \
	cmath_cpow.c \
	cmath_lpshiftfreq.c \
	cmath_catan.c
lib@GSTPLUGINPREFIX@cmath_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
lib@GSTPLUGINPREFIX@cmath_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
lib@GSTPLUGINPREFIX@cmath_la_LIBADD = $(top_builddir)/lib/libgstlalcalibration.la
lib@GSTPLUGINPREFIX@cmath_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

# headers we need but don't want installed
noinst_HEADERS = cmath_base.h
