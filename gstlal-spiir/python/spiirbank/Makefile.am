# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

spiirbankdir = $(pkgpythondir)/spiirbank

spiirbank_PYTHON = \
	__init__.py \
	spiir_utils.py \
	spiir_decomp.py \
	cbc_template_iir.py \
	optimizer.py

spiirbank_LTLIBRARIES = _spiir_decomp.la optimizer_cy.la

_spiir_decomp_la_SOURCES = _spiir_decomp.c
_spiir_decomp_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS) -DMODULE_NAME="\"gstlal.spiirbank._spiir_decomp\""
_spiir_decomp_la_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(GSL_CFLAGS) -fno-strict-aliasing -DMODULE_NAME="\"gstlal.spiirbank._spiir_decomp\""
_spiir_decomp_la_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(GSL_LIBS) $(PYTHON_LIBS) -module -avoid-version -llalinspiral

optimizer_cy.c: optimizer_cy.pyx
	cython $^

optimizer_cy_la_SOURCES = optimizer_cy.c
optimizer_cy_la_CPPFLAGS = $(AM_CPPFLAGS) $(PYTHON_CPPFLAGS)
optimizer_cy_la_CFLAGS = $(AM_CFLAGS) $(PYTHON_CFLAGS) -fno-strict-aliasing
optimizer_cy_la_LDFLAGS = $(AM_LDFLAGS) $(PYTHON_LIBS) -module -avoid-version -lm
