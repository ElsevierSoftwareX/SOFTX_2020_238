#SUBDIRS = datasource \
#	  postcoh \
#	  spiir

AM_CPPFLAGS = -I$(top_srcdir)/lib

# This is a trick taken from the gst-python automake setup.
# All of the Python scripts will be installed under the exec dir,
# which prevents the module from getting spread across lib and lib64
# on e.g. CentOS.
pkgpythondir = $(pkgpyexecdir)

pkgpython_PYTHON = \
	cbc_template_iir.py \
	postcoh_finalsink.py \
	uni_datasource.py \
	cohfar.py \
	postcoh_table_def.py \
	ligolw_inspinjfind_postcoh.py \
	optimize_mf.py \
	spiirparts.py 
