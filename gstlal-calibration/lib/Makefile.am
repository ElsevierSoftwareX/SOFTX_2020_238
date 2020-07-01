AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

pkginclude_HEADERS = gstlal_firtools.h
pkgconfig_DATA = gstlal-calibration.pc
lib_LTLIBRARIES = libgstlalcalibration.la

libgstlalcalibration_la_SOURCES = gstlal_firtools.h gstlal_firtools.c
libgstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
libgstlalcalibration_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
