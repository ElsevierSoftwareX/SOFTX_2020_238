\page gstlalinspiralcodeoptimization gstlal inspiral code optimization

WARNING: This will make pretty nonportable code.  E.g., if you build on a
Xeon-XXXX that might be the only processor it will run on.

My environment looks like this:

	# This is an install prefix that does not get used anywhere but this script
	INSTALLPATH=/home/gstlalcbc/profilegcc

	export CFLAGS="-fPIC -O3 -march=native"

	# These are environment variables that do get exported
	PATH=${INSTALLPATH}/bin:$PATH
	PKG_CONFIG_PATH=${INSTALLPATH}/lib64/pkgconfig:${INSTALLPATH}/lib/pkgconfig:$PKG_CONFIG_PATH
	PYTHONPATH=${INSTALLPATH}/lib64/python2.6/site-packages:${INSTALLPATH}/lib/python2.6/site-packages:$PYTHONPATH:/usr/lib/python2.6/site-packages:/usr/lib64/python2.6/site-packages/
	GST_PLUGIN_PATH=${INSTALLPATH}/lib/gstreamer-0.10:/opt/lscsoft/gst/lib64/gstreamer-0.10

	export PATH PKG_CONFIG_PATH PYTHONPATH GST_PLUGIN_PATH

\section deps Install dependencies

All dependencies are available here:

https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/deps/

you need to build in this order:

 -# atlas
 -# orc
 -# gsl
 -# gstreamer
 -# gst-plugins-base
 -# gst-plugins-good
 -# gst-python
 -# oprofile (not critical but nice to see how things are working)


\subsection atlas atlas special instructions

Follow build instructions, but configure with

	--nof77 --shared

Add the following line to your environment once ATLAS is built

        export LD_PRELOAD=${INSTALLPATH}/lib/libsatlas.so:${INSTALLPATH}/lib/libtatlas.so

You may want to log out and back in again to resource your environment

You will also need to build lalsuite and gstlal in the normal way, but with the CFLAGS above

\section profiling Profiling

 You can find a self contained profile script here: https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/profile.tar.gz

NOTE: This only works with the released gstlal versions, e.g., gstlal-0.7.1.tar.gz and gstlal-inspiral-0.3.2.tar.gz
