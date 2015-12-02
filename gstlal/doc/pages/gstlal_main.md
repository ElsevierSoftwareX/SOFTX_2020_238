\mainpage Welcome

## Getting Started

- You can get a development copy of the gstlal software suite from git.  Follow the <a href="https://www.lsc-group.phys.uwm.edu/daswg/docs/howto/advanced-lalsuite-git.html#clone">instructions for lalsuite</a>, but use "https://versions.ligo.org/git/lalsuite.git" for the repository URL.

- Release tar balls and some binary packages are available <a href=https://www.lsc-group.phys.uwm.edu/daswg/download/repositories.html>here</a>.

- Build and install from source.  This follows the normal GNU build procedures involving 1) ./00init.sh 2) ./configure 3) make 4) make install.  You should build the packages in order of gstlal, gstlal-ugly, gstlal-calibration, gstlal-inspiral.  If you are building to a non FHS place (e.g., your home directory) you will need to ensure some environment variables are set so that your installation will function.  How to do this correct (what paths to use, etc.) is left as an exercise to the user, but the following five variables must be set, and the we show examples of what they should be set to ("${PREFIX}" is the path used for --prefix when configure was run)

		GI_TYPELIB_PATH="${PREFIX}/lib/girepository-1.0:${GI_TYPELIB_PATH}"
		GST_PLUGIN_PATH="${PREFIX}/lib/gstreamer-0.10:${GST_PLUGIN_PATH}"
		PATH="${PREFIX}/bin:${PATH}"
		# Debian systems need lib, RH systems need lib64, including both doesn't hurt
		PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH}"
		# Debian systems need lib, RH systems need lib and lib64
		PYTHONPATH="${PREFIX}/lib64/python2.6/site-packages:${PREFIX}/lib/python2.6/site-packages:$PYTHONPATH"

		export GI_TYPELIB_PATH GST_PLUGIN_PATH PATH PKG_CONFIG_PATH PYTHONPATH

## Documentation for gstlal elements

- <a href="@gstlalgtkdoc/">See here for more details</a>

## Making fake data

- \ref gstlalfakedataoverviewpage
- Relevant programs
  - \ref gstlal_fake_frames
  - \ref gstlal_fake_frames_pipe

## Measuring PSDs

### Relevant programs

  - \ref gstlal_reference_psd
  - \ref gstlal_plot_psd
  - \ref gstlal_plot_psd_horizon
  - \ref gstlal_psd_polyfit
  - \ref gstlal_psd_xml_from_asd_txt

## Data interaction

### Relevant programs

  - \ref gstlal_spectrum_movie
  - \ref gstlal_play

## Profiling
  - \ref gstlalo1profiling

## References

- <a href=@gstlalinspiraldoc> gstlal_inspiral documentation</a>
- <a href=http://gstreamer.freedesktop.org/> gstreamer home page </a>
- \ref gstlal_review_main_page
- \ref gstlalmeetingspage
- \ref gstlalteleconspage
