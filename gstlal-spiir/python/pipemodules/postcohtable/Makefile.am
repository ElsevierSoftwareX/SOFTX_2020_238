ADD_CFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/lib/include
ADD_LIBS = -L\$(top_srcdir)/lib -lgstlalspiir


# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

postcohtabledir = $(pkgpythondir)/pipemodules/postcohtable
postcohtable_PYTHON = \
	__init__.py \
	postcoh_table_def.py \
	postcohtable.py 
	
postcohtable_LTLIBRARIES = _postcohtable.la

_postcohtable_la_SOURCES = _postcohtable.c
_postcohtable_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal.pipemodules.postcohtable._postcohtable\""
_postcohtable_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(gstreamer_CFLAGS) $(GSTLAL_CFLAGS) $(ADD_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal.pipemodules.postcohtable._postcohtable\""
_postcohtable_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(GSTLAL_LDFLAGS) $(PYTHON_LIBS) $(ADD_LIBS) -module -avoid-version

