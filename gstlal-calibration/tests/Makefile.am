# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	test_common.py

EXTRA_DIST = \
	L-L1_TEST_FRAME-10-10.gwf \
	lal_logical_undersampler_test_01.py

#TESTS = lal_logical_undersampler_test_01.py

clean-local :
	rm -f *.dump
	rm -f *.avi
	rm -f *.pyc
