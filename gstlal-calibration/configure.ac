#
# Preamble
#


AC_INIT([gstlal-calibration],[1.2.12],[gstlal-discuss@ligo.org],[gstlal-calibration])
AC_COPYRIGHT([Copyright (C) 2008-2019 Madeline Wade, Aaron Viets])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])
AM_INIT_AUTOMAKE([1.9 tar-ustar foreign])
AUTOMAKE_OPTIONS="${AUTOMAKE_OPTIONS} --add-missing --copy"
AC_CONFIG_FILES([ \
	Makefile \
	gstlal-calibration.spec \
	bin/Makefile \
	debian/control \
	debian/Makefile \
	lib/Makefile \
	gst/Makefile \
	gst/lal/Makefile \
	gst/cmath/Makefile \
	gst/splitcounter/Makefile \
	python/Makefile \
	share/Makefile \
	tests/Makefile \
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


AC_SUBST([LIBAPI], [0])
AC_SUBST([LIBREL], [0])
AC_SUBST([LIBAGE], [0])
AC_SUBST([LIBVERSION], [${LIBAPI}:${LIBREL}:${LIBAGE}])


#
# Check for programs
#

# check for c99 compiler
m4_pattern_allow([AC_PROG_CC_C99])
m4_ifdef([AC_PROG_CC_C99],[AC_PROG_CC_C99],[LALSUITE_AC_PROG_CC_C99])

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
AC_SUBST([MIN_PKG_CONFIG_VERSION], [0.18.0])
PKG_PROG_PKG_CONFIG("$MIN_PKG_CONFIG_VERSION")


#
# Check for Python
#


MIN_PYTHON_VERSION="2.7"
AM_PATH_PYTHON(["$MIN_PYTHON_VERSION"],,)
AX_PYTHON_DEVEL()
# hack to remove default lib dirs from PYTHON_LIBS.  only tested on Debian
# and SL6.  FIXME:  find a way to do this properly
PYTHON_LIBS=[`echo "$PYTHON_LIBS" | sed -e "s?-L/usr/lib[^[:space:]]*??g"`]
AC_SUBST([MIN_PYTHON_VERSION])
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
# Check for GStreamer
#


AC_SUBST([GSTREAMER_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_VERSION], [1.14.1])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-controller-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your default gstreamer plugin search path, you might need to add the directory

	$plugindir
	
to your GST_PLUGIN_PATH environment variable.])
AC_SUBST([GSTLAL_PLUGIN_LDFLAGS], ["-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\|gstlal_\|GstLAL_\|GSTLAL_\).*"])

gstreamer_VERSION=`$PKG_CONFIG --modversion gstreamer-${GSTREAMER_RELEASE}`
AX_COMPARE_VERSION([$gstreamer_VERSION], [ge], [1.14.0], [
	GSTPLUGINPREFIX="gst"
], [
	GSTPLUGINPREFIX=""
])
AC_SUBST([GSTPLUGINPREFIX])


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
# Check for GStreamer Audio Library
#


AC_SUBST([GSTREAMER_AUDIO_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_AUDIO_VERSION], [1.2.4])
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
# Check for LAL
#


AC_SUBST([MIN_LAL_VERSION], [6.15.2])
AC_SUBST([MIN_LALMETAIO_VERSION], [1.2.6])
PKG_CHECK_MODULES([LAL], [lal >= ${MIN_LAL_VERSION} lalmetaio >= ${MIN_LALMETAIO_VERSION} lalsupport])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])


#
# Check for ligo_segments
#


AC_SUBST([MIN_LIGO_SEGMENTS_VERSION], [1.1.0])
AX_PYTHON_LIGO_SEGMENTS([$MIN_LIGO_SEGMENTS_VERSION])


#
# Check for gstlal
#


AC_SUBST([MIN_GSTLAL_VERSION], [1.0.0])
AC_SUBST([MIN_GSTLALUGLY_VERSION], [1.5.6])
AC_SUBST([MIN_GSTPLUGINSMATH_VERSION], [1.0.1])
PKG_CHECK_MODULES([GSTLAL], [gstlal >= ${MIN_GSTLAL_VERSION}])
AC_SUBST([GSTLAL_VERSION], [`$PKG_CONFIG --modversion gstlal`])
AX_GSTLAL_SPLIT_VERSION([GSTLAL_VERSION])
AC_SUBST([GSTLAL_VERSION_MAJOR])
AC_SUBST([GSTLAL_VERSION_MINOR])
AC_SUBST([GSTLAL_VERSION_POINT])
AC_SUBST([GSTLAL_CFLAGS])
AC_SUBST([GSTLAL_LIBS])


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
