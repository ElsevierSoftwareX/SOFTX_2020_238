\mainpage gstlalmainpage Welcome page

\section gstlalmainpagegettingstarted Getting Started

- You can get a development copy of the gstlal software suite from git via:

		$ git clone albert.einstein@ligo-vcs.phys.uwm.edu:/usr/local/git/gstlal

- Release tar balls and some binary packages are available <a href=https://www.lsc-group.phys.uwm.edu/daswg/download/repositories.html>here</a>.

- Installation.  This follows the normal GNU buildsystem procedures involving 1) ./00init.sh 2) ./configure 3) make 4) make install.  You should build the packages in order of gstlal, gstlal-ugly, gstlal-calibration, gstlal-inspiral.  If you ar building to a non FHS place (e.g. your home directory) you will need something like the following environment to be set before building and when using the software.  Please make sure your environment is sane and cruft free otherwise.

		# This is an install prefix that does not get used anywhere but this script, it is not exported !!!
		INSTALLPATH=/home/channa/gstlocal

		# These are environment variables that do get exported
		PATH=${INSTALLPATH}/bin:$PATH
		PKG_CONFIG_PATH=${INSTALLPATH}/lib64/pkgconfig:${INSTALLPATH}/lib/pkgconfig:$PKG_CONFIG_PATH
		PYTHONPATH=${INSTALLPATH}/lib64/python2.6/site-packages:${INSTALLPATH}/lib/python2.6/site-packages:$PYTHONPATH
		GST_PLUGIN_PATH=${INSTALLPATH}/lib/gstreamer-0.10:/opt/lscsoft/gst/lib64/gstreamer-0.10

		export PATH PKG_CONFIG_PATH PYTHONPATH GST_PLUGIN_PATH

### Making fake data

- \ref gstlalfakedataoverviewpage
- Relevant programs
  - gstlal_fake_frames
  - gstlal_fake_frames_pipe

### Measuring PSDs

- Overview
- Relevant programs
  - gstlal_reference_psd
  - gstlal_plot_psd
  - gstlal_plot_psd_horizon

### Data interaction

- Overview
- Relevant programs
  - gstlal_spectrum_movie
  - gstlal_play

### References
- <a href=http://gstreamer.freedesktop.org/> gstreamer home page </a>
- \ref gstlalmeetingspage
