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
	glib-genmarshal --header --prefix gstlal_marshal $(srcdir)/$< >$@.tmp && mv -f $@.tmp $@

gstlal_marshal.c : gstlal_marshal.list
	glib-genmarshal --body --prefix gstlal_marshal $(srcdir)/$< >$@.tmp && mv -f $@.tmp $@

#
# Build gstlal_peakfinder.h.  Note that it's not portable to process files
# ending in something other than ".c" with the C preprocessor, so just to
# be sure the source for the header fragment is named ".c".  we need to
# make sure gstlal_peakfinder.h gets built before gstlal_peakfinder.c is
# compiled, hence the extra rule for that
#

gstlal_peakfinder.c : gstlal_peakfinder.h

gstlal_peakfinder_types.h : gstlal_peakfinder_types.c
	$(CPP) -C -P -o $@ $<

gstlal_peakfinder.h : gstlal_peakfinder_top.h gstlal_peakfinder_types.h gstlal_peakfinder_bottom.h
	cat $+ >$@

clean-local :
	rm -f gstlal_marshal.c gstlal_marshal.h
	rm -f gstlal_peakfinder.h gstlal_peakfinder_types.h
