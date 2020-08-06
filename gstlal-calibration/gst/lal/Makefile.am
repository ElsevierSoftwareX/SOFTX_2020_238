AM_CPPFLAGS = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@gstlalcalibration.la

lib@GSTPLUGINPREFIX@gstlalcalibration_la_SOURCES = \
	gstlalcalibration.c \
	gstlal_add_constant.c gstlal_add_constant.h \
	gstlal_wings.c gstlal_wings.h \
	gstlal_complexfirbank.c gstlal_complexfirbank.h \
	gstlal_smoothcalibfactors.c gstlal_smoothcalibfactors.h \
	gstlal_smoothkappas.c gstlal_smoothkappas.h \
	gstlal_constantupsample.c gstlal_constantupsample.h \
	gstlal_resample.c gstlal_resample.h \
	gstlal_logicalundersample.c gstlal_logicalundersample.h \
	gstlal_demodulate.c gstlal_demodulate.h \
	gstlal_insertgap.c gstlal_insertgap.h \
	gstlal_fccupdate.c gstlal_fccupdate.h \
	gstlal_transferfunction.c gstlal_transferfunction.h \
	gstlal_trackfrequency.c gstlal_trackfrequency.h \
	gstlal_adaptivefirfilt.c gstlal_adaptivefirfilt.h \
	gstlal_dqtukey.c gstlal_dqtukey.h \
	gstlal_property.c gstlal_property.h \
	gstlal_typecast.c gstlal_typecast.h \
	gstlal_matrixsolver.c gstlal_matrixsolver.h \
	gstlal_sensingtdcfs.c gstlal_sensingtdcfs.h \
	gstlal_makediscont.c gstlal_makediscont.h \
	gstlal_randreplace.c gstlal_randreplace.h \
	gstlal_detectchange.c gstlal_detectchange.h
lib@GSTPLUGINPREFIX@gstlalcalibration_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
lib@GSTPLUGINPREFIX@gstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
lib@GSTPLUGINPREFIX@gstlalcalibration_la_LIBADD = $(top_builddir)/lib/libgstlalcalibration.la
lib@GSTPLUGINPREFIX@gstlalcalibration_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
