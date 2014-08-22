\page gstlalinspiralcodeoptimization gstlal inspiral code optimization

\section Introduction

This documentation page covers installation of optimized dependencies for use in gstlal profiling and benchmarking.  The instructions below cover a general case, however at the end there are links to pages for detailed instruction sets for specific machines.  After reading this page, the user should be well prepared to install all of the needed software on a vanilla machine and to then perform profiling and benchmarking.

\section Install Makefile

A makefile, the corresponding rc file to source, and required patches are available here:

https://ldas-jobs.ligo.caltech.edu/~cody.messick/vanilla_make/

A description of the patches can be found in the \ref gstlalinspiralpatchinformation page. 

Running this makefile (with the correct environment variables from optimalrc of course) with 

	make -f Makefile.ligosoftware

will download tarballs and install 

 - atlas 3.10.1
 - orc 0.4.18
 - gsl 1.15
 - gstreamer 0.10.36
 - gst-plugins-base 0.10.36
 - gst-plugins-good 0.10.31
 - gst-python 0.10.22
 - oprofile 0.9.9
 - libframe 8.21
 - metaio 8.3.0
 - swig 2.0.11
 - ldas-tools 2.0.0
 - lal 6.12.0
 - lalframe 1.1.1
 - lalmetaio 1.2.0
 - lalsimulation 1.1.0
 - lalinspiral 1.5.2
 - lalburst 1.2.2
 - laldetchar 0.1.0
 - lalpulsar 1.9.0
 - lalinference 1.3.1
 - lalstochastic 1.1.10
 - lalapps 6.14.0
 - pylal 0.5.0
 - glue 1.46
 - gstlal 0.7.1
 - gstlal-ugly 0.6.0
 - gstlal-calibration 0.1.1
 - gstlal-inspiral 0.3.2

Before running the Makefile the following dependencies, available on both yum and apt, need to be installed with their development files (ie the dev packages are needed).

 - libxml2
 - pygtk-2.0
 - libpopt
 - binutils
 - pcre
 - matplotlib
 - scipy
 - openssl
 - fftw

Please email cody.messick@ligo.org if you find you need additional required packages.

There are a couple of FIXMEs listed in the Makefile, which are duplicated below.

	## FIXME --enable-gcc-flags set to no on lalapps configure as work around to avoid warnings stopping install process
	## Not sure what causes the warnings
	$(INSTALL_FILES_DIR)/lalapps-6.14.0/config.log : $(INSTALL_DIR)/lib/liblalstochastic.so
		tar -xzf $(TARDIR)/lalapps-6.14.0.tar.gz -C $(INSTALL_FILES_DIR)
		cd $(INSTALL_FILES_DIR)/lalapps-6.14.0 && \
			./configure --enable-gcc-flags=no --enable-swig-python --prefix=$(INSTALL_DIR)

	...

	## FIXME Hack to make gsl default to atlas for blas
	$(DEP_INSTALL_DIR)/lib/pkgconfig/gsl.pc.orig : $(DEP_INSTALL_DIR)/lib/libgsl.so
			cp $(DEP_INSTALL_DIR)/lib/pkgconfig/gsl.pc $@ 
				sed 's/-lgslcblas/-latlas -lsatlas/' $@ > $(DEP_INSTALL_DIR)/lib/pkgconfig/gsl.pc

\section Install by hand

WARNING: This will make pretty nonportable code.  E.g., if you build on a
Xeon-XXXX that might be the only processor it will run on.

Set up an environment similar to below

	# This is an install prefix that does not get used anywhere but this script
	INSTALLPATH=/home/gstlalcbc/profilegcc

	export CFLAGS="-fPIC -O3 -march=native"

	# These are environment variables that do get exported
	PATH=${INSTALLPATH}/bin:$PATH
	PKG_CONFIG_PATH=${INSTALLPATH}/lib64/pkgconfig:${INSTALLPATH}/lib/pkgconfig:$PKG_CONFIG_PATH
	PYTHONPATH=${INSTALLPATH}/lib64/python2.6/site-packages:${INSTALLPATH}/lib/python2.6/site-packages:$PYTHONPATH:/usr/lib/python2.6/site-packages:/usr/lib64/python2.6/site-packages/
	GST_PLUGIN_PATH=${INSTALLPATH}/lib/gstreamer-0.10:/opt/lscsoft/gst/lib64/gstreamer-0.10

	export PATH PKG_CONFIG_PATH PYTHONPATH GST_PLUGIN_PATH

Dependencies locations are linked below.  You need to build in this order, but first read below for special atlas and gsl install instructions.

 -# <a href="http://downloads.sourceforge.net/project/math-atlas/Stable/3.10.1/atlas3.10.1.tar.bz2">atlas</a>
 -# <a href="http://gstreamer.freedesktop.org/src/orc/orc-0.4.18.tar.gz">orc</a>
 -# <a href="http://ftp.gnu.org/gnu/gsl/gsl-1.15.tar.gz">gsl</a>
 -# <a href="http://gstreamer.freedesktop.org/src/gstreamer/gstreamer-0.10.36.tar.gz">gstreamer</a>
 -# <a href="http://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-0.10.36.tar.gz">gst-plugins-base</a>
 -# <a href="http://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-0.10.31.tar.gz">gst-plugins-good</a>
 -# <a href="http://gstreamer.freedesktop.org/src/gst-python/gst-python-0.10.22.tar.gz">gst-python</a>
 -# <a href="http://prdownloads.sourceforge.net/oprofile/oprofile-0.9.9.tar.gz">oprofile</a>


After unpacking atlas from the tarball, follow the build directions but configure with

	--nof77 --shared

After installing gsl, open the file installdirectory/lib/pkgconfig/gsl.pc.  Replace the line

	-lgslcblas

with 

	-latlas -lsatlas

This will force gsl to go through atlas when using BLAS.  Note that this is a hacky workaround because gsl has problems compiling to use atlas correctly.

After the dependencies listed above are installed, build lalsuite and gstlal in the normal way, but with the CFLAGS above

\section profiling Profiling

 You can find a self contained profile script here: https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/profile.tar.gz

NOTE: This only works with the released gstlal versions, e.g., gstlal-0.7.1.tar.gz and gstlal-inspiral-0.3.2.tar.gz
