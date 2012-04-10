pkginclude_HEADERS = gstlal.h gstlal_debug.h gstlal_tags.h gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstaudioadapter.h gstlalcollectpads.h gstlal_peakfinder.h gstlal_autocorrelation_chi2.h
pkgconfig_DATA = gstlal.pc
lib_LTLIBRARIES = libgstlal.la libgstlaltags.la libgstlaltypes.la

libgstlal_la_SOURCES = gstlal.c gstlal.h gstlal_debug.h gstlal_cdf_weighted_chisq_P.c gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstlal_segments.c gstlal_peakfinder.h gstlal_peakfinder.c gstlal_autocorrelation_chi2.h gstlal_autocorrelation_chi2.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(gstreamer_LIBS)

libgstlaltags_la_SOURCES = gstlal_tags.c gstlal_tags.h
libgstlaltags_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstlaltags_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(gstreamer_LIBS)

libgstlaltypes_la_SOURCES = gstlal_debug.h gstaudioadapter.h gstaudioadapter.c gstlalcollectpads.h gstlalcollectpads.c
libgstlaltypes_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstlaltypes_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(gstreamer_LIBS)
