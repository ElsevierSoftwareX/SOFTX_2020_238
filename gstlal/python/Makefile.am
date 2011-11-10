INCLUDES = -I$(top_srcdir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)
# Put these Python scripts into the plugin directory under "python" so that
# GStreamer sees them.
#pkgpythondir = $(plugindir)/python
#pkgpyexecdir = $(pkgpythondir)

pkgpython_PYTHON = \
	__init__.py \
	elements.py \
	lloidparts.py \
	matplotlibhelper.py \
	misc.py \
	pipeio.py \
	pipeparts.py \
	pipeutil.py \
	reference_psd.py \
	simulation.py

pkgpyexec_LTLIBRARIES = _misc.la

_misc_la_SOURCES = misc.c
_misc_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
_misc_la_LIBADD = $(top_builddir)/lib/libgstlal.la
_misc_la_LDFLAGS = $(AM_LDFLAGS) -module $(PYTHON_LDFLAGS)
