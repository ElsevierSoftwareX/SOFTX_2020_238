# Put these Python scripts into the plugin directory under "python" so that
# GStreamer sees them.
# FIXME: elements must be byte-compiled with python 3.  remove when
# configure.ac is picking up python3 for us
PYTHON = $(shell env which python3)
pkgpythondir = $(plugindir)/python
pkgpyexecdir = $(pkgpythondir)

#pkgpython_PYTHON = lal_channelgram.py lal_checktimestamps.py lal_fakeadvvirgosrc.py lal_fakeadvligosrc.py lal_fakeligosrc.py lal_histogramplot.py lal_lho_coherent_null.py lal_spectrumplot.py
pkgpython_PYTHON = lal_checktimestamps.py lal_spectrumplot.py
