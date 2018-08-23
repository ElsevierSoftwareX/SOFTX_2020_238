SUBDIRS = postcohtable

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pipemodulesdir = $(pkgpythondir)/pipemodules

pipemodules_PYTHON = \
	__init__.py \
	homomorphic.py \
	uni_datasource.py \
	postcoh_finalsink.py \
	spiirparts.py \
	pipe_macro.py
