plugin_LTLIBRARIES = libgstlalcalibration.la

libgstlalcalibration_la_SOURCES = \
	gstlalcalibration.c \
	gstlal_add_constant.c gstlal_add_constant.h \
	gstlal_pyfuncsrc.c gstlal_pyfuncsrc.h \
	gstlal_wings.c gstlal_wings.h \
	gstlal_smoothcalibfactors.c gstlal_smoothcalibfactors.h \
	gstlal_constantupsample.c gstlal_constantupsample.h \
	gstlal_logicalundersample.c gstlal_logicalundersample.h \
	gstlal_demodulate.c gstlal_demodulate.h
libgstlalcalibration_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
libgstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
libgstlalcalibration_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LDFLAGS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
