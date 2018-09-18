plugin_LTLIBRARIES = libsplitcounter.la

libsplitcounter_la_SOURCES = \
	splitcounter.c splitcounter.h
libsplitcounter_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
libsplitcounter_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS)
libsplitcounter_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSTLAL_LIBS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
