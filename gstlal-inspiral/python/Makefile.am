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
	far.py \
	hoftcache.py \
	inspiral_pipe.py \
	inspiral.py \
	lloidparts.py \
	lloidplots.py \
	streamthinca.py \
	svd_bank.py \
	templates.py
