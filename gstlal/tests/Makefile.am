# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

EXTRA_DIST = \
	cachesrc_test_01.sh \
	cmp_nxydumps.py \
	firbank_test_01.py \
	framesrc_test_01.py \
	gate_test_01.py \
	lal_reblock_test_01.sh \
	matrixmixer_test_01.py \
	resample_test_01.py \
	segmentsrc_test_01.py \
	statevector_test_01.py \
	sumsquares_test_01.py \
	togglecomplex_test_01.py \
	whiten_test_01.kst \
	whiten_test_01.py \
	test_common.py

TESTS = cachesrc_test_01.sh firbank_test_01.py gate_test_01.py lal_reblock_test_01.sh matrixmixer_test_01.py resample_test_01.py segmentsrc_test_01.py statevector_test_01.py sumsquares_test_01.py togglecomplex_test_01.py whiten_test_01.py

pkgpython_PYTHON = \
	cmp_nxydumps.py

clean-local :
	rm -vf *.dump
	rm -vf *.avi
	rm -vf *.pyc
