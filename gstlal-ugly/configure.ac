#
# Preamble
#


AC_INIT([gstlal-ugly],[0.7.3],[gstlal-discuss@ligo.org],[gstlal-ugly])
AC_COPYRIGHT([Copyright (C) 2008--2014 Kipp Cannon])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])
AM_INIT_AUTOMAKE([1.9 tar-ustar foreign])
AUTOMAKE_OPTIONS="${AUTOMAKE_OPTIONS} --add-missing --copy"
AC_CONFIG_FILES([ \
	Makefile \
	gstlal-ugly.spec \
	debian/control \
	debian/Makefile \
	python/Makefile \
	share/Makefile \
	gst/Makefile \
	gst/nds/Makefile \
	gst/lal/Makefile \
	gst/python/Makefile \
	gst/framecpp/Makefile \
	gst/gds/Makefile \
	gst/multirate/Makefile \
	gst/cuda/Makefile \
	bin/Makefile \
	tests/Makefile \
	examples/Makefile
])


#
# Split package version
#


AX_GSTLAL_SPLIT_VERSION([PACKAGE_VERSION])
AC_SUBST([PACKAGE_VERSION_MAJOR])
AC_SUBST([PACKAGE_VERSION_MINOR])
AC_SUBST([PACKAGE_VERSION_POINT])


#
# Move headers into a versioned sub-directory to allow more than one set of
# headers to be installed system-wide.  (commented out until this is
# needed)
#


#AC_SUBST([includedir],["\${includedir}/\${PACKAGE}-\${PACKAGE_VERSION}"])


#
# Extra directories
#


AC_SUBST([docdir], ["\${datadir}/doc"])
AC_SUBST([pkgdocdir], ["\${docdir}/\${PACKAGE_NAME}-\${PACKAGE_VERSION}"])
AC_SUBST([pkgconfigdir],["\${libdir}/pkgconfig"])


#
# force /usr/share/ package files into same directory as gstlal
#


AC_SUBST([pkgdatadir],["\${datadir}/gstlal"])


#
# Set the library API info
#
#  0.0.0   Original version
#  1.0.0   0.2.0 release (sngl_burst.h api changed)


AC_SUBST([LIBAPI], [1])
AC_SUBST([LIBREL], [0])
AC_SUBST([LIBAGE], [0])
AC_SUBST([LIBVERSION], [${LIBAPI}:${LIBREL}:${LIBAGE}])


#
# Check for programs
#

# check for c99 compiler
m4_pattern_allow([AC_PROG_CC_C99])
m4_ifdef([AC_PROG_CC_C99],[AC_PROG_CC_C99],[LALSUITE_AC_PROG_CC_C99])

AC_PROG_CXX
AC_PROG_INSTALL
# libtool incorrectly determines library path on SL6.  FIXME:  remove when
# no longer needed
case `cat /etc/redhat-release 2> /dev/null` in
  "Scientific Linux"*|"CentOS"*)
    AC_MSG_NOTICE([hacking round broken libtool multilib support on RedHat systems])
    lt_cv_sys_lib_dlsearch_path_spec="/lib64 /usr/lib64"
    ;;
esac
LT_INIT
PKG_PROG_PKG_CONFIG()


#
# Check for Python
#


AM_PATH_PYTHON([2.6],,)
AX_PYTHON_DEVEL()
# hack to remove default lib dirs from PYTHON_LDFLAGS.  only tested on
# Debian and SL6.  FIXME:  find a way to do this properly
PYTHON_LDFLAGS="`echo $PYTHON_LDFLAGS | sed -e 'sX-L/usr/lib\(64\)\?\(/\)\?[[:space:]]*XXg'`"
# hack to add missing options to PYTHON_LDFLAGS.  sigh, what's the point of
# AX_PYTHON_DEVEL!?
PYTHON_LDFLAGS="-module -avoid-version $PYTHON_LDFLAGS"
# force python modules into same directory as gstlal
AC_SUBST([pkgpythondir], ["\${pythondir}/gstlal"])
AC_SUBST([pkgpyexecdir], ["\${pyexecdir}/gstlal"])


#
# CFLAGS
#


AX_CFLAGS_WARN_ALL([AM_CFLAGS])
AM_CFLAGS="$AM_CFLAGS -Wextra -Wno-missing-field-initializers -Wno-unused-parameter"	# extra gcc-specific stuff
AC_SUBST([AM_CFLAGS])


#
# LDFLAGS
#


# Turn on error messages for undefined symbols
AM_LDFLAGS="$AM_LDFLAGS -no-undefined"
AC_SUBST([AM_LDFLAGS])


#
# Check for math library
#


AC_CHECK_LIB([m], [main], , [AC_MSG_ERROR([Not found!])])


#
# Check for FFTW
#


PKG_CHECK_MODULES([FFTW], [fftw3 fftw3f])
AC_SUBST([FFTW_CFLAGS])
AC_SUBST([FFTW_LIBS])
AC_DEFINE([GSTLAL_FFTW_WISDOM_ENV], ["GSTLAL_FFTW_WISDOM"], [Set to the name of the environment variable to use for overriding the system-wide double-precision FFTW wisdom file])
AC_DEFINE([GSTLAL_FFTWF_WISDOM_ENV], ["GSTLAL_FFTWF_WISDOM"], [Set to the name of the environment variable to use for overriding the system-wide single-precision FFTW wisdom file])


#
# Check for NDS2
#


AC_SUBST([MIN_NDS_VERSION], [0.11.2])
AC_ARG_WITH(
	[nds],
	[AS_HELP_STRING([--with-nds], [include NDS source element @<:@default=check@:>@])],
	[],
	[with_nds=check]
)
AS_IF([test "x$with_nds" != "xno"], [
	PKG_CHECK_MODULES(
		[NDS],
		[nds2-client >= ${MIN_NDS_VERSION}],
		[AC_DEFINE([HAVE_NDS], [1], [Define if you have nds2-client])
		HAVE_NDS="yes"],
		[HAVE_NDS="no"
		AS_IF([test "x$with_nds" != "xcheck"], [
			AC_MSG_ERROR([Not found!])
		], [
			AC_MSG_WARN([Not found!])
		])
	])
])
AM_CONDITIONAL([COND_NDS], [test "x${HAVE_NDS}" == "xyes"])
AC_SUBST([NDS_CFLAGS])
AC_SUBST([NDS_LIBS])


#
# Check for GStreamer
#


AC_SUBST([GSTREAMER_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_VERSION], [0.10.32])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your default gstreamer plugin search path, you might need to add the directory

	$plugindir
	
to your GST_PLUGIN_PATH environment variable.])
AC_SUBST([GSTLAL_PLUGIN_LDFLAGS], ["-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\|gstlal_\|GstLAL_\|GSTLAL_\).*"])

# Check for availability of GstBaseParse class (starting in 0.10.33)
CFLAGS_saved="$CFLAGS"
CFLAGS="$gstreamer_CFLAGS"
AC_COMPILE_IFELSE(
	[AC_LANG_PROGRAM([#include <gst/base/gstbaseparse.h>])],
	[HAVE_GSTBASEPARSE=yes
	AC_DEFINE([HAVE_GST_BASEPARSE], [1], [Define if GstBaseParse class is available.])],
	[HAVE_GSTBASEPARSE=no]
)
CFLAGS="$CFLAGS_saved"
AM_CONDITIONAL([COND_GSTBASEPARSE], [test "x${HAVE_GSTBASEPARSE}" == "xyes"])


#
# Check for GStreamer Video Library
#


AC_SUBST([GSTREAMER_VIDEO_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_VIDEO_VERSION], [0.10.32])
PKG_CHECK_MODULES([gstreamer_video], [gstreamer-video-${GSTREAMER_VIDEO_RELEASE} >= ${MIN_GSTREAMER_VIDEO_VERSION}])
AC_SUBST([gstreamer_video_CFLAGS])
AC_SUBST([gstreamer_video_LIBS])


#
# Check for GStreamer Audio Library
#


AC_SUBST([GSTREAMER_AUDIO_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_AUDIO_VERSION], [0.10.32])
PKG_CHECK_MODULES([gstreamer_audio], [gstreamer-audio-${GSTREAMER_AUDIO_RELEASE} >= ${MIN_GSTREAMER_AUDIO_VERSION}])
AC_SUBST([gstreamer_audio_CFLAGS])
AC_SUBST([gstreamer_audio_LIBS])


#
# Check for NumPy
#


AX_PYTHON_MODULE(numpy, fatal)
NUMPY_CFLAGS=-I`$PYTHON -c "import numpy;print (numpy.get_include());"`
old_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS ${PYTHON_CPPFLAGS} ${NUMPY_CFLAGS}"
AC_CHECK_HEADER([numpy/arrayobject.h],
	[PYTHON_CPPFLAGS="${PYTHON_CPPFLAGS} ${NUMPY_CFLAGS}"],
	[AC_MSG_ERROR([Numpy extension header not found])],
	[#include "Python.h"])
CFLAGS="$old_CFLAGS"


#
# Check for framecpp
#
# FIXME:  this isn't really optional any more.  make its absence an error
#


AC_SUBST([MIN_FRAMECPP_VERSION], [2.0.4])
AC_ARG_WITH([framecpp],
	[AS_HELP_STRING([--with-framecpp], [include framecpp plugin @<:@default=check@:>@])],
	[],
	[with_framecpp=check]
)
AS_IF([test "x$with_framecpp" != "xno"], [
	PKG_CHECK_MODULES(
		[framecpp],
		[framecpp >= ${MIN_FRAMECPP_VERSION}],
		[HAVE_FRAMECPP="yes"],
		[AC_MSG_WARN([Not found: $framecpp_PKG_ERRORS])
		HAVE_FRAMECPP="no"]
	)
], [
	AC_MSG_WARN([check for framecpp disabled])
	HAVE_FRAMECPP="no"
])
AM_CONDITIONAL([COND_FRAMECPP], [test "x${HAVE_FRAMECPP}" == "xyes"])
AS_IF([test "x${HAVE_FRAMECPP}" == "xyes"], [
	AC_DEFINE([HAVE_FRAMECPP], [1], [Defined if you have framecpp])
	framecpp_VERSION=`$PKG_CONFIG --modversion framecpp`
	AC_MSG_NOTICE([framecpp version: $framecpp_VERSION])
	AX_COMPARE_VERSION([$framecpp_VERSION], [ge], [2.0.0], [
		AC_DEFINE([HAVE_FRAMECPP_2x], [1], [Defined if framecpp >= 2.x])
	])
	# FIXME:  the "-std=c++0x" option is here as a temporary hack until
	# we can find a way to get the autoconf macros to configure the
	# cxxflags for us.  note that in later versions of g++ the flag
	# should be -std=c++11 (c++0x is deprecated)
	framecpp_CXXFLAGS="${framecpp_CFLAGS} -std=c++0x -D__STDC_CONSTANT_MACROS"
])
AC_SUBST([framecpp_CFLAGS])
AC_SUBST([framecpp_CXXFLAGS])
AC_SUBST([framecpp_LIBS])
AC_SUBST([framecpp_VERSION])


#
# Check for GDS
#


AC_SUBST([MIN_GDS_VERSION], [2.16.16])
AC_ARG_WITH(
	[gds],
	[AS_HELP_STRING([--with-gds], [include gds plugin @<:@default=check@:>@])],
	[],
	[with_gds=check]
)
AS_IF([test "x$with_gds" != "xno"], [
	PKG_CHECK_MODULES(
		[gds],
		[gds >= ${MIN_GDS_VERSION}],
		[HAVE_GDS="yes"
		AC_DEFINE([HAVE_GDS], [1], [Defined if you have gds])
		gds_LIBS="$gds_LIBS -llsmp"],	# hack to work around broken .pc file
		[AC_MSG_WARN([Not found: $gds_PKG_ERRORS])
		HAVE_GDS="no"]
	)
], [
	AC_MSG_WARN([check for gds disabled])
	HAVE_GDS="no"
])
AM_CONDITIONAL([COND_GDS], [test "x${HAVE_GDS}" == "xyes"])
AC_SUBST([gds_CFLAGS])
AC_SUBST([gds_LIBS])
AS_IF([test "x${HAVE_GDS}" == "xyes"], [
	AC_SUBST([framexmit_CFLAGS], ["-I/usr/include/gds"])
	AC_SUBST([framexmit_LIBS], ["-lframexmit -lpthread -lsockutil"])
])


#
# Check for LAL
#


AC_SUBST([MIN_LAL_VERSION], [6.13.0])
AC_SUBST([MIN_LALMETAIO_VERSION], [1.2.0])
PKG_CHECK_MODULES([LAL], [lal >= ${MIN_LAL_VERSION} lalmetaio >= ${MIN_LALMETAIO_VERSION} lalsupport])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])


#
# Check for glue and pylal
#


AC_SUBST([MIN_GLUE_VERSION], [1.47])
AC_SUBST([MIN_PYLAL_VERSION], [0.6.0])
AX_PYTHON_GLUE()
AX_PYTHON_PYLAL()


#
# Check for gstlal
#


AC_SUBST([MIN_GSTLAL_VERSION], [0.8.0])
PKG_CHECK_MODULES([GSTLAL], [gstlal >= ${MIN_GSTLAL_VERSION}])
AC_SUBST([GSTLAL_VERSION], [`$PKG_CONFIG --modversion gstlal`])
AX_GSTLAL_SPLIT_VERSION([GSTLAL_VERSION])
AC_SUBST([GSTLAL_VERSION_MAJOR])
AC_SUBST([GSTLAL_VERSION_MINOR])
AC_SUBST([GSTLAL_VERSION_POINT])
AC_SUBST([GSTLAL_CFLAGS])
AC_SUBST([GSTLAL_LIBS])


#
# Check for GSL
#


PKG_CHECK_MODULES([GSL], [gsl])
AC_SUBST([GSL_CFLAGS])
AC_SUBST([GSL_LIBS])


#
# Check for ORC
#


PKG_CHECK_MODULES([ORC], [orc-0.4], , [
	AC_MSG_WARN([ORC not found;  using potentially slower code paths.])
	ORC_CFLAGS="-DDISABLE_ORC"
])
AC_SUBST([ORC_CFLAGS])
AC_SUBST([ORC_LIBS])


#
# Check for CUDA
#


AC_ARG_WITH(
	[cuda],
	[AS_HELP_STRING([--with-cuda], [specify location of CUDA [/usr/local/cuda]])],
	[],
	[with_cuda=no]
)
AS_IF([test "x$with_cuda" != "xno"], [
	AS_IF([test "x$build_cpu" = "xx86_64"], [
		CLIBS="lib64"
	], [
		CLIBS="lib"
	])
	AC_SUBST([NVCC], ["${with_cuda}/bin/nvcc"])
	AC_SUBST([NVCC_CFLAGS], ["-I${with_cuda}/include"])
	PKG_CHECK_MODULES([GLIB], [glib-2.0])
	AC_SUBST([NVCC_GSTLAL_CFLAGS], [$GLIB_CFLAGS])
	AC_SUBST([NVCC_gstreamer_CFLAGS], ["$($PKG_CONFIG --cflags-only-I gstreamer-0.10)"])
	AC_SUBST([NVCC_LIBS], ["-L${with_cuda}/$CLIBS -lcudart"])
	HAVE_CUDA="yes"
], [
	AC_MSG_WARN([check for cuda disabled])
	HAVE_CUDA="no"
])
AC_DEFINE([HAVE_CUDA], [test "x${HAVE_CUDA}" == "xyes"], [Defined if you have cuda])
AM_CONDITIONAL([COND_CUDA], [test "x${HAVE_CUDA}" == "xyes"])


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
