SUBDIRS = skymap

pkginclude_HEADERS = gstlal.h gstlal_tags.h gstlal_cdf_weighted_chisq_P.h
pkgconfig_DATA = gstlal.pc
lib_LTLIBRARIES = libgstlal.la libgstlaltags.la

libgstlal_la_SOURCES = gstlal.c gstlal.h gstlal_cdf_weighted_chisq_P.c gstlal_cdf_weighted_chisq_P.h
libgstlal_la_CPPFLAGS = $(GSL_CLFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LIBADD = $(gstreamer_LIBS) $(LAL_LIBS) $(GSL_LIBS) $(FFTW_LIBS)
libgstlal_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LDFLAGS) $(LAL_LDFLAGS) $(gstreamer_LDFLAGS)

libgstlaltags_la_SOURCES = gstlal_tags.c gstlal_tags.h
libgstlaltags_la_CPPFLAGS = $(gstreamer_CFLAGS)
libgstlaltags_la_LIBADD = $(gstreamer_LIBS)
libgstlaltags_la_LDFLAGS =  $(gstreamer_LDFLAGS)
