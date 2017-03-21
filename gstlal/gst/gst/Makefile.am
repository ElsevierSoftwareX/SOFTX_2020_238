AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

plugin_LTLIBRARIES = libgstlaladder.la libgstpythonplugin.la

libgstlaladder_la_SOURCES = \
	gstadder.h gstadder.c \
	gstadderorc.h gstadderorc-dist.h gstadderorc-dist.c
libgstlaladder_la_CFLAGS = $(AM_CFLAGS) $(ORC_CFLAGS) $(gstreamer_CFLAGS) $(gstreamer_audio_CFLAGS) -DGST_PACKAGE_NAME=\"gstlal\" -DGST_PACKAGE_ORIGIN=\"\"
libgstlaladder_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlal.la $(top_builddir)/lib/gstlal/libgstlaltypes.la
libgstlaladder_la_LDFLAGS = $(AM_LDFLAGS) $(ORC_LIBS) $(gstreamer_LIBS) $(gstreamer_audio_LIBS) $(GSTLAL_PLUGIN_LDFLAGS)

libgstpythonplugin_la_SOURCES = gstpythonplugin.c
libgstpythonplugin_la_CFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) $(PYGOBJECT_CFLAGS) $(gstreamer_CFLAGS) $(GST_PLUGINS_BASE_CFLAGS) -DPYTHON_VERSION="\"$(PYTHON_VERSION)\"" -DPY_LIB_LOC="\"$(PYTHON_LIB_LOC)\"" -DPY_ABI_FLAGS="\"$(PYTHON_ABI_FLAGS)\"" -DPY_LIB_SUFFIX="\"$(PYTHON_LIB_SUFFIX)\"" -DGST_API_VERSION="\"$(GSTREAMER_RELEASE)\"" -DPLUGINDIR="\"$(plugindir)\""
libgstpythonplugin_la_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_LIBS) $(gstreamer_LIBS) $(GSTLAL_PLUGIN_LDFLAGS) $(PYGOBJECT_LIBS)
