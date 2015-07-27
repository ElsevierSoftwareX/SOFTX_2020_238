\mainpage Welcome page

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
		GST_PLUGIN_PATH=${INSTALLPATH}/lib/gstreamer-0.10:${INSTALLPATH}/lib64/gstreamer-0.10:${GST_PLUGIN_PATH}

		export PATH PKG_CONFIG_PATH PYTHONPATH GST_PLUGIN_PATH

### Documentation for gstlal elements

- <a href="@gstlalgtkdoc/">See here for more details</a>

### Making fake data

- \ref gstlalfakedataoverviewpage
- Relevant programs
  - \ref gstlal_fake_frames
  - \ref gstlal_fake_frames_pipe

### Measuring PSDs

- Relevant programs
  - \ref gstlal_reference_psd
  - \ref gstlal_plot_psd
  - \ref gstlal_plot_psd_horizon
  - \ref gstlal_psd_polyfit
  - \ref gstlal_psd_xml_from_asd_txt

### Data interaction

- Relevant programs
  - \ref gstlal_spectrum_movie
  - \ref gstlal_play

### References
- <a href=@gstlalinspiraldoc> gstlal_inspiral documentation</a>
- <a href=http://gstreamer.freedesktop.org/> gstreamer home page </a>
- \ref gstlal_review_main_page
- \ref gstlalteleconspage
