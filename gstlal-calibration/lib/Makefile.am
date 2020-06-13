pkginclude_HEADERS = gstlal_firtools.h
pkgconfig_DATA = gstlal-calibration.pc
lib_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalcalibration.la

lib@GSTPLUGINPREFIX@gstlalcalibration_la_SOURCES = gstlal_firtools.c gstlal_firtools.h
lib@GSTPLUGINPREFIX@gstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS)
lib@GSTPLUGINPREFIX@gstlalcalibration_la_LDFLAGS = -version-info $(LIBVERSION) $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS)
