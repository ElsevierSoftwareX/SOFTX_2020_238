SUBDIRS = pipeparts stats

AM_CPPFLAGS = -I$(top_srcdir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	__init__.py \
	bottle.py \
	coherent_null.py \
	dagfile.py \
	dagparts.py \
	datasource.py \
	elements.py \
	httpinterface.py \
	matplotlibhelper.py \
	misc.py \
	multirate_datasource.py \
	pipeio.py \
	pipeline.py \
	pipeutil.py \
	plotpsd.py \
	plotutil.py \
	reference_psd.py \
	servicediscovery.py \
	simplehandler.py \
	simulation.py

pkgpyexec_LTLIBRARIES = _misc.la

_misc_la_SOURCES = misc.c
_misc_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
_misc_la_LIBADD = $(top_builddir)/lib/gstlal/libgstlal.la
_misc_la_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_LIBS) -module -avoid-version
