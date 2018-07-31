# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

workflowdir = $(pkgpythondir)/workflow

workflow_PYTHON = \
	__init__.py \
	dbtables_postcoh.py \
	ligolw_inspinjfind_postcoh.py \
	ligolw_sqlite_postcoh.py \
	utils.py
