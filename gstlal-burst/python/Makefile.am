AM_CPPFLAGS = -I$(top_srcdir)/lib

SUBDIRS = excesspower snax

pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = snglbursttable.py

pkgpyexec_LTLIBRARIES = _snglbursttable.la

_snglbursttable_la_SOURCES = snglbursttable.c snglburstrowtype.h
_snglbursttable_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._snglbursttable\""
_snglbursttable_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(gstreamer_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._snglbursttable\""
_snglbursttable_la_LIBADD = $(top_builddir)/lib/libgstlalburst.la
_snglbursttable_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version
