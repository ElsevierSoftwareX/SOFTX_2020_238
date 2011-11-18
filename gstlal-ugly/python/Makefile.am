# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)
# Put these Python scripts into the plugin directory under "python" so that
# GStreamer sees them.
#pkgpythondir = $(plugindir)/python
#pkgpyexecdir = $(pkgpythondir)

pkgpython_PYTHON = \
	__init__.py \
	cbc_template_fir.py \
	cbc_template_iir.py \
	dagparts.py \
	svd_bank.py \
	templates.py \
	far.py \
	streamthinca.py \
	omega.py \
	lloidplots.py \
	inspiral.py \
	excesspower.py
