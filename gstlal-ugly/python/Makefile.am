#AM_CPPFLAGS = -I$(top_srcdir)/gst/framecpp
#AM_CPPFLAGS = -I$(top_srcdir)/lib
AM_CPPFLAGS = -I$(top_srcdir)/gst/framecpp -I$(top_srcdir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	aggregator.py \
	metric.py \
	tree.py

pkgpyexec_LTLIBRARIES =
