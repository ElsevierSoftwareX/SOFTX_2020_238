SUBDIRS = elements
INCLUDES = -I$(top_srcdir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	__init__.py \
	cbc_template_fir.py \
	cbc_template_iir.py \
	dagparts.py \
	lloidparts.py \
	lloidplots.py \
	matplotlibhelper.py \
	misc.py \
	namedtuple.py \
	option.py \
	pipeio.py \
	pipeparts.py \
	pipeutil.py \
	reference_psd.py \
	svd_bank.py \
	templates.py \
	ligolw_output.py

pkgpyexec_LTLIBRARIES = _misc.la

_misc_la_SOURCES = misc.c
_misc_la_CPPFLAGS = $(PYTHON_CPPFLAGS)
_misc_la_LDFLAGS = $(AM_LDFLAGS) -module $(PYTHON_LDFLAGS)
_misc_la_LIBADD = $(top_builddir)/lib/libgstlal.la
