Installation
===============

You can get a development copy of the gstlal software suite from git.  Doing this at minimum will require a development copy of lalsuite.
  * https://git.ligo.org/lscsoft/gstlal
  * https://git.ligo.org/lscsoft/lalsuite

Source tarballs for GstLAL packages and all the LIGO/Virgo software dependencies are available here: http://software.ligo.org/lscsoft/source/

Limited binary packages are available here: https://wiki.ligo.org/Computing/DASWG/SoftwareDownloads 

Building and installing from source follows the normal GNU build procedures
involving:

 1. ./00init.sh 
 2. ./configure 
 3. make 
 4. make install.

You should build the packages in order of gstlal, gstlal-ugly,
gstlal-calibration, gstlal-inspiral.  If you are building to a non FHS place
(e.g., your home directory) you will need to ensure some environment variables
are set so that your installation will function.  The following five variables
must be set.  As **just an example**::

	GI_TYPELIB_PATH="/path/to/your/installation/lib/girepository-1.0:${GI_TYPELIB_PATH}"
	GST_PLUGIN_PATH="/path/to/your/installation/lib/gstreamer-0.10:${GST_PLUGIN_PATH}"
	PATH="/path/to/your/installation/bin:${PATH}"
	# Debian systems need lib, RH systems need lib64, including both doesn't hurt
	PKG_CONFIG_PATH="/path/to/your/installation/lib/pkgconfig:/path/to/your/installation/lib64/pkgconfig:${PKG_CONFIG_PATH}"
	# Debian systems need lib, RH systems need lib and lib64
	PYTHONPATH="/path/to/your/installation/lib64/python2.7/site-packages:/path/to/your/installation/lib/python2.7/site-packages:$PYTHONPATH"

