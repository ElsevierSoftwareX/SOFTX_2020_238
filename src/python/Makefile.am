SUBDIRS = elements
INCLUDES = -I$(top_srcdir)/src/lib

pkgpython_PYTHON = __init__.py cbc_template_fir.py dagparts.py misc.py pipeio.py pipeparts.py templates.py

pkgpyexec_LTLIBRARIES = _misc.la

_misc_la_SOURCES = misc.c
_misc_la_CPPFLAGS = $(PYTHON_CPPFLAGS)
_misc_la_LDFLAGS = -module $(PYTHON_LDFLAGS)
_misc_la_LIBADD = $(top_srcdir)/src/lib/libgstlal.la
