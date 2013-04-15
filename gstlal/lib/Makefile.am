EXTRA_DIST =
CLEANFILES =

pkginclude_HEADERS = gstlal.h gstlal_debug.h gstlal_marshal.h gstlal_tags.h gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstaudioadapter.h gstlalcollectpads.h gstlal_peakfinder.h gstlal_autocorrelation_chi2.h gstlal_gps_clock.h
pkgconfig_DATA = gstlal.pc
lib_LTLIBRARIES = libgstlal.la libgstlaltags.la libgstlaltypes.la

libgstlal_la_SOURCES = gstlal.c gstlal.h gstlal_debug.h gstlal_marshal.c gstlal_marshal.h gstlal_cdf_weighted_chisq_P.c gstlal_cdf_weighted_chisq_P.h gstlal_segments.h gstlal_segments.c gstlal_peakfinder.h gstlal_peakfinder.c gstlal_autocorrelation_chi2.h gstlal_autocorrelation_chi2.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) $(LAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlal_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(GSL_LIBS) $(LAL_LIBS) $(gstreamer_LIBS)

libgstlaltags_la_SOURCES = gstlal_tags.c gstlal_tags.h
libgstlaltags_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstlaltags_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(gstreamer_LIBS)

libgstlaltypes_la_SOURCES = gstlal_debug.h gstaudioadapter.h gstaudioadapter.c gstlalcollectpads.h gstlalcollectpads.c gstlal_gps_clock.h gstlal_gps_clock.c
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
INTROSPECTION_SCANNER_ARGS = --add-include-path=$(srcdir) --identifier-prefix GstLAL
INTROSPECTION_COMPILER_ARGS = --includedir=$(srcdir)

if HAVE_INTROSPECTION
gstlaltypes.gir : libgstlaltypes.la Makefile
gstlaltypes_gir_INCLUDES = Gst-0.10
gstlaltypes_gir_CFLAGS = $(INCLUDES)
gstlaltypes_gir_LIBS = libgstlaltypes.la
gstlaltypes_gir_FILES = gstlal_gps_clock.h gstlal_gps_clock.c
gstlaltypes_gir_NAMESPACE = gstlaltypes
gstlaltypes_gir_VERSION = $(VERSION)
INTROSPECTION_GIRS += gstlaltypes.gir

girdir = $(datadir)/gir-1.0
gir_DATA = $(INTROSPECTION_GIRS)

typelibdir = $(libdir)/girepository-1.0
typelib_DATA = $(INTROSPECTION_GIRS:.gir=.typelib)

CLEANFILES += $(gir_DATA) $(typelib_DATA)
endif
