#
# Preamble
#


AC_INIT([gstlal-inspiral],[0.2.3],[gstlal-discuss@ligo.org],[gstlal-inspiral])
AC_COPYRIGHT([Copyright (C) 2008--2013 Kipp Cannon])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])
AM_INIT_AUTOMAKE([1.9 tar-ustar foreign])
AUTOMAKE_OPTIONS="${AUTOMAKE_OPTIONS} --add-missing --copy"
AC_CONFIG_FILES([ \
	Makefile \
	gstlal-inspiral.spec \
	debian/control \
	debian/Makefile \
	lib/gstlal-inspiral.pc \
	lib/Makefile \
	lib/skymap/Makefile \
	python/Makefile \
	python/emcee/Makefile \
	gst/Makefile \
	gst/lal/Makefile \
	gst/python/Makefile \
	bin/Makefile \
	share/Makefile \
	tests/Makefile
	doc/Doxyfile \
	doc/Makefile
])


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
AC_PATH_PROG(DOXYGEN, doxygen)
AM_CONDITIONAL([COND_DOXYGEN], [test -n "${DOXYGEN}"])

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
# Check for GStreamer
#


AC_SUBST([GSTREAMER_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_VERSION], [0.10.32])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your gstreamer's default search path, you might need to add the directory

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
# Check for PyGObject
#


PKG_CHECK_MODULES([pygobject], [pygobject-2.0])
AC_SUBST([pygobject_CFLAGS])
AC_SUBST([pygobject_LIBS])


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


AC_SUBST([MIN_LAL_VERSION], [6.11.0])
AC_SUBST([MIN_LALMETAIO_VERSION], [1.1.1])
AC_SUBST([MIN_LALINSPIRAL_VERSION], [1.5.1])
PKG_CHECK_MODULES([LAL], [lal >= ${MIN_LAL_VERSION} lalmetaio >= ${MIN_LALMETAIO_VERSION} lalsupport lalinspiral >= ${MIN_LALINSPIRAL_VERSION}])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])


#
# Check for glue and pylal
#


AC_SUBST([MIN_GLUE_VERSION], [1.45])
AC_SUBST([MIN_PYLAL_VERSION], [0.4.0])
AX_PYTHON_GLUE()
AX_PYTHON_PYLAL()


#
# Check for gstlal
#


AC_SUBST([MIN_GSTLAL_VERSION], [0.6.0])
AC_SUBST([MIN_GSTLALUGLY_VERSION], [0.5.0])
PKG_CHECK_MODULES([GSTLAL], [gstlal >= ${MIN_GSTLAL_VERSION}])
AC_SUBST([GSTLAL_VERSION], [`$PKG_CONFIG --modversion gstlal`])
AC_SUBST([GSTLAL_CFLAGS])
AC_SUBST([GSTLAL_LIBS])


#
# Check for GSL
#


PKG_CHECK_MODULES([GSL], [gsl])
AC_SUBST([GSL_CFLAGS])
AC_SUBST([GSL_LIBS])


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
