INCLUDES = -I$(top_srcdir)/lib

# Put these Python scripts into the plugin directory under "python" so that
# GStreamer sees them.
pkgpythondir = $(plugindir)/python
pkgpyexecdir = $(pkgpythondir)

pkgpython_PYTHON = lal_checktimestamps.py lal_fakeadvligosrc.py lal_fakeligosrc.py lal_fakeskymapsrc.py lal_faketriggersrc.py lal_onlinehoftsrc.py lal_numpy_functiongenerator.py lal_timeseriesplotter.py lal_skymaprenderer.py

pkgpyexec_LTLIBRARIES = _onlinehoftsrc.la

_onlinehoftsrc_la_SOURCES = _onlinehoftsrc.c
_onlinehoftsrc_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(pygobject_CFLAGS)
_onlinehoftsrc_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(pygobject_LIBS)
