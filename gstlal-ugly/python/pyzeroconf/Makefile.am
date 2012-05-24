# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)/pyzeroconf

pkgpython_PYTHON = __init__.py Zeroconf.py

EXTRA_DIST = Browser.py lgpl.txt README ZeroconfTest.py ZeroconfTest2.py
