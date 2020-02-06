AM_CPPFLAGS = -I$(abs_srcdir)/..

EXTRA_DIST =
CLEANFILES =

pkginclude_HEADERS = ezligolw.h ezxml.h gstlal.h gstlal_debug.h gstlal_marshal.h gstlal_tags.h gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstaudioadapter.h gstlalcollectpads.h gstlal_peakfinder.h gstlal_autocorrelation_chi2.h gstlal_gps_clock.h gstlal_frhistory.h gstlal_audio_info.h
pkgconfig_DATA = gstlal.pc
lib_LTLIBRARIES = libgstlal.la libgstlaltags.la libgstlaltypes.la

libgstlal_la_SOURCES = ezligolw.c ezligolw.h ezxml.c ezxml.h gstlal.c gstlal.h gstlal_debug.h gstlal_fftw.c gstlal_marshal.c gstlal_marshal.h gstlal_cdf_weighted_chisq_P.c gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstlal_segments.c gstlal_peakfinder.h gstlal_peakfinder.c gstlal_autocorrelation_chi2.h gstlal_autocorrelation_chi2.c gstlal_audio_info.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(FFTW_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(FFTW_LIBS) $(GSL_LIBS) $(LAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS)

libgstlaltags_la_SOURCES = gstlal_tags.c gstlal_tags.h
libgstlaltags_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstlaltags_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(gstreamer_LIBS)

libgstlaltypes_la_SOURCES = gstlal_debug.h gstaudioadapter.h gstaudioadapter.c gstlalcollectpads.h gstlalcollectpads.c gstlal_gps_clock.h gstlal_gps_clock.c gstlal_frhistory.h gstlal_frhistory.c
libgstlaltypes_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlaltypes_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(gstreamer_LIBS)

#
# Build gstlal_marshal.h, gstlal_marshal.c
#

gstlal_marshal.h : gstlal_marshal.list
	glib-genmarshal --header --prefix gstlal_marshal $< >$@.tmp && mv -f $@.tmp $@

gstlal_marshal.c : gstlal_marshal.list
	glib-genmarshal --body --prefix gstlal_marshal $< >$@.tmp && mv -f $@.tmp $@

EXTRA_DIST += gstlal_marshal.list
CLEANFILES += gstlal_marshal.c gstlal_marshal.h

#
# Build gstlal_peakfinder.h.  Note that it's not portable to process files
# ending in something other than ".c" with the C preprocessor, so just to
# be sure the source for the header fragment is named ".c".  we need to
# make sure gstlal_peakfinder.h gets built before gstlal_peakfinder.c is
# compiled, hence the extra rule for that
#

gstlal_peakfinder.c : gstlal_peakfinder.ct gstlal_peakfinder.h

gstlal_peakfinder_types.h : gstlal_peakfinder_types.c gstlal_peakfinder.ht
	$(CPP) -C -P -o $@ $<

gstlal_peakfinder.h : gstlal_peakfinder_top.h gstlal_peakfinder_types.h gstlal_peakfinder_bottom.h
	cat $+ >$@

EXTRA_DIST += gstlal_peakfinder_types.c gstlal_peakfinder.ht gstlal_peakfinder.ct gstlal_peakfinder_top.h gstlal_peakfinder_bottom.h
CLEANFILES += gstlal_peakfinder.h gstlal_peakfinder_types.h

#
# gobject introspection
#

-include $(INTROSPECTION_MAKEFILE)
INTROSPECTION_GIRS =
INTROSPECTION_SCANNER_ARGS = --warn-all --add-include-path=$(srcdir) --identifier-prefix GstLAL --identifier-prefix gstlal_frhistory
INTROSPECTION_COMPILER_ARGS = --includedir=$(srcdir)
dist_gir_DATA =
typelib_DATA =

if HAVE_INTROSPECTION
INTROSPECTION_GIRS += GstLAL-$(GIR_VERSION).gir
GstLAL-$(GIR_VERSION).gir : libgstlaltypes.la Makefile
# automake won't let me use variables on the LHS of the following for some
# reason, so I've resorted to using autoconf substitution macros.  ugly.
GstLAL_@GIR_VERSION_@_gir_INCLUDES = Gst-1.0
GstLAL_@GIR_VERSION_@_gir_CFLAGS = $(INCLUDES)
GstLAL_@GIR_VERSION_@_gir_LIBS = libgstlaltypes.la
GstLAL_@GIR_VERSION_@_gir_FILES = gstlal_gps_clock.h gstlal_gps_clock.c gstlal_frhistory.h gstlal_frhistory.c
GstLAL_@GIR_VERSION_@_gir_NAMESPACE = GstLAL
GstLAL_@GIR_VERSION_@_gir_VERSION = $(GIR_VERSION)

dist_gir_DATA += $(INTROSPECTION_GIRS)
typelib_DATA += $(INTROSPECTION_GIRS:.gir=.typelib)
CLEANFILES += $(dist_gir_DATA) $(typelib_DATA)
endif
