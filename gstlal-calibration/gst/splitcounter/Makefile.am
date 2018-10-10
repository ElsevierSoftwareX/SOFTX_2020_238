plugin_LTLIBRARIES = lib@GSTPLUGINPREFIX@splitcounter.la

lib@GSTPLUGINPREFIX@splitcounter_la_SOURCES = \
	splitcounter.c splitcounter.h
lib@GSTPLUGINPREFIX@splitcounter_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
lib@GSTPLUGINPREFIX@splitcounter_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
lib@GSTPLUGINPREFIX@splitcounter_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
