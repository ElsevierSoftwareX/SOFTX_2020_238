INCLUDES = -I$(top_srcdir)/lib

lib_LTLIBRARIES = libgstframecpp.la

libgstframecpp_la_SOURCES = \
	gstfrhistory.c gstfrhistory.h
libgstframecpp_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
libgstframecpp_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS)

plugin_LTLIBRARIES = libframecpp_plugin.la

libframecpp_plugin_la_SOURCES = \
	framecpp_plugin.c \
	gstfrpad.c gstfrpad.h \
	marshal.c marshal.h \
	muxqueue.c muxqueue.h \
	muxcollectpads.c muxcollectpads.h \
	framecpp_channelmux.cc framecpp_channelmux.h \
	framecpp_channeldemux.cc framecpp_channeldemux.h \
	framecpp_filesink.c framecpp_filesink.h
if COND_GSTBASEPARSE
libframecpp_plugin_la_SOURCES += framecpp_igwdparse.cc framecpp_igwdparse.h
endif
libframecpp_plugin_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) $(framecpp_CFLAGS)
# FIXME:  the "-std=c++0x" option is here as a temporary hack until we can
# find a way to get the autoconf macros to configure the cflags for us
# correctly on its own
libframecpp_plugin_la_CXXFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) $(framecpp_CFLAGS) -std=c++0x -D__STDC_CONSTANT_MACROS
libframecpp_plugin_la_LDFLAGS = $(AM_LDFLAGS) $(GSTLAL_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(framecpp_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)
libframecpp_plugin_la_LIBADD = libgstframecpp.la
