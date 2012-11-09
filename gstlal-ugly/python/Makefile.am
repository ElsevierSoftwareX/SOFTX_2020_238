SUBDIRS = pyzeroconf

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	epparts.py \
	excesspower.py \
	gviz_api.py \
	httpinterface.py \
	omega.py \
	servicediscovery.py
