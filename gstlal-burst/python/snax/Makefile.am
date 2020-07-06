AM_CPPFLAGS = -I$(top_srcdir)/lib -I$(top_builddir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)
snaxdir = $(pkgpythondir)/snax

snax_PYTHON = \
	__init__.py \
	auxcache.py \
	utils.py \
	waveforms.py \
	feature_extractor.py \
	multichannel_datasource.py \
	pipeparts.py \
	sngltriggertable.py

pkgpyexec_LTLIBRARIES = _sngltriggertable.la

_sngltriggertable_la_SOURCES = sngltriggertable.c
_sngltriggertable_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal._sngltriggertable\""
_sngltriggertable_la_CFLAGS = $(AM_CFLAGS) $(GSTLAL_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) $(gstreamer_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal._sngltriggertable\""
_sngltriggertable_la_LIBADD = $(top_builddir)/lib/gstlal-burst/libgstlalburst.la
_sngltriggertable_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version
