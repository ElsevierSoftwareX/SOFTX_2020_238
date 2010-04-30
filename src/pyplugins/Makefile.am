INCLUDES = -I$(top_srcdir)/src/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(plugindir)/python

pkgpython_PYTHON = lal_whitehoftsrc.py
