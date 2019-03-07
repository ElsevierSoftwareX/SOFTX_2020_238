AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

pkginclude_HEADERS = gstlal_snglburst.h gstlal_sngltrigger.h sngltriggerrowtype.h
pkgconfig_DATA = gstlal-burst.pc
lib_LTLIBRARIES = libgstlalburst.la

libgstlalburst_la_SOURCES = \
	gstlal_snglburst.h gstlal_snglburst.c \
	gstlal_sngltrigger.h gstlal_sngltrigger.c \
	sngltriggerrowtype.h sngltriggerrowtype.c
libgstlalburst_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalburst_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
