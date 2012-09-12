SUBDIRS = pyzeroconf

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
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
	inspiral_pipe.py \
	excesspower.py \
	gviz_api.py \
	httpinterface.py \
	servicediscovery.py
