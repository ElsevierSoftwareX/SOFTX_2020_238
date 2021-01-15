AM_CPPFLAGS = -I$(top_srcdir)/lib

SUBDIRS = config dags emcee plots stats

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	cbc_template_fir.py \
	chirptime.py \
	far.py \
	hoftcache.py \
	ilwdify.py \
	imr_utils.py \
	inspiral_pipe.py \
	inspiral.py \
	lloidhandler.py \
	lloidparts.py \
	lvalert_helper.py \
	rate_estimation.py \
	snglinspiraltable.py \
	spawaveform.py \
	streamthinca.py \
	svd_bank.py \
	svd_bank_snr.py \
	templates.py \
	p_astro_gstlal.py

pkgpyexec_LTLIBRARIES = _rate_estimation.la _snglinspiraltable.la _spawaveform.la

_rate_estimation_la_SOURCES = rate_estimation.c
_rate_estimation_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._rate_estimation\""
_rate_estimation_la_CFLAGS = $(AM_CFLAGS) $(GSL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._rate_estimation\""
_rate_estimation_la_LDFLAGS = $(AM_LDFLAGS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version

_snglinspiraltable_la_SOURCES = snglinspiraltable.c
_snglinspiraltable_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._snglinspiraltable\""
_snglinspiraltable_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(gstreamer_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._snglinspiraltable\""
_snglinspiraltable_la_LIBADD = $(top_builddir)/lib/libgstlalinspiral.la
_snglinspiraltable_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version

_spawaveform_la_SOURCES = _spawaveform.c
_spawaveform_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._spawaveform\""
_spawaveform_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._spawaveform\""
_spawaveform_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version

