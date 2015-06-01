AM_CPPFLAGS = -I$(top_srcdir)/lib

SUBDIRS = emcee

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	cbc_template_fir.py \
	cbc_template_iir.py \
	chirptime.py \
	curve_fit.py \
	far.py \
	hoftcache.py \
	inspiral_pipe.py \
	inspiral.py \
	lloidparts.py \
	lloidplots.py \
	llweb.py \
	plotfar.py \
	plotsegments.py \
	rate_estimation.py \
	spiirparts.py \
	streamthinca.py \
	svd_bank.py \
	templates.py

pkgpyexec_LTLIBRARIES = _rate_estimation.la

_rate_estimation_la_SOURCES = rate_estimation.c
_rate_estimation_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._rate_estimation\""
_rate_estimation_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._rate_estimation\""
_rate_estimation_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(PYTHON_LDFLAGS) -module -avoid-version
