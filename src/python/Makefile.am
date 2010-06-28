SUBDIRS = elements
INCLUDES = -I$(top_srcdir)/src/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = __init__.py cbc_template_fir.py dagparts.py misc.py namedtuple.py pipeio.py pipeparts.py pipeutil.py templates.py

pkgpyexec_LTLIBRARIES = _misc.la

_misc_la_SOURCES = misc.c
_misc_la_CPPFLAGS = $(PYTHON_CPPFLAGS)
_misc_la_LDFLAGS = -module $(PYTHON_LDFLAGS)
_misc_la_LIBADD = $(top_srcdir)/src/lib/libgstlal.la
