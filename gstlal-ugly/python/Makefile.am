AM_CPPFLAGS = -I$(top_srcdir)/gst/framecpp

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	epparts.py \
	excesspower.py \
	gviz_api.py \
	httpinterface.py \
	servicediscovery.py \
	vetosrc.py

pkgpyexec_LTLIBRARIES =

if COND_FRAMECPP
pkgpyexec_LTLIBRARIES += _gstfrhistory.la

_gstfrhistory_la_SOURCES = _gstfrhistory.c
_gstfrhistory_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
_gstfrhistory_la_CFLAGS = $(AM_CFLAGS) $(gstreamer_CFLAGS)
_gstfrhistory_la_LIBADD = $(top_builddir)/gst/framecpp/libgstframecpp.la
_gstfrhistory_la_LDFLAGS = $(AM_LDFLAGS) $(gstreamer_LIBS) $(PYTHON_LDFLAGS) -module -avoid-version
endif
