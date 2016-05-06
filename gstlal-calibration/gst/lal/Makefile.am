plugin_LTLIBRARIES = libgstlalcalibration.la

libgstlalcalibration_la_SOURCES = \
	gstlalcalibration.c \
        gstlal_wings.c gstlal_wings.h 
libgstlalcalibration_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
libgstlalcalibration_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
