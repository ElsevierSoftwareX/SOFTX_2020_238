INCLUDES = -I$(top_srcdir)/lib

plugin_LTLIBRARIES = libframecpp_plugin.la

libframecpp_plugin_la_SOURCES = \
	framecpp_plugin.c \
	gstfrpad.c gstfrpad.h \
	marshal.c marshal.h \
	muxqueue.c muxqueue.h \
	muxcollectpads.c muxcollectpads.h \
	framecpp_channelmux.cc framecpp_channelmux.h \
	framecpp_channeldemux.cc framecpp_channeldemux.h
if COND_GSTBASEPARSE
libframecpp_plugin_la_SOURCES += framecpp_igwdparse.cc framecpp_igwdparse.h
endif
libframecpp_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_CXXFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) $(framecpp_CFLAGS)
libframecpp_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(framecpp_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
