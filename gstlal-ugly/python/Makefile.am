AM_CPPFLAGS = -I$(top_srcdir)/gst/framecpp -I$(top_srcdir)/gst/cuda/postcoh

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	gviz_api.py \
	httpinterface.py \
	servicediscovery.py \
	spiirparts.py \
	uni_datasource.py \
	postcoh_table_def.py \
	postcoh_finalsink.py \
	cohfar.py \
	ligolw_inspinjfind_postcoh.py \
	homomorphic.py \
	postcohtable.py \
	spawaveform.py


pkgpyexec_LTLIBRARIES = _spawaveform.la _postcohtable.la

_postcohtable_la_SOURCES = postcohtable.c
_postcohtable_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._postcohtable\""
_postcohtable_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(gstreamer_CFLAGS) $(GSTLAL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._postcohtable\""
_postcohtable_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(GSTLAL_LDFLAGS) $(PYTHON_LIBS) -module -avoid-version

_spawaveform_la_SOURCES = _spawaveform.c
_spawaveform_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._spawaveform\""
_spawaveform_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._spawaveform\""
_spawaveform_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version

