#
# Preamble
#


AC_INIT([gstlal],[1.5.0],[gstlal-discuss@ligo.org],[gstlal])
AC_COPYRIGHT([Copyright (C) The authors (see source code for details)])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])
AM_INIT_AUTOMAKE([1.9 tar-ustar foreign])
AUTOMAKE_OPTIONS="${AUTOMAKE_OPTIONS} --add-missing --copy"
AC_CONFIG_FILES([ \
	Makefile \
	gstlal.spec \
	debian/control \
	debian/Makefile \
	lib/Makefile \
	lib/gstlal/gstlal.pc \
	lib/gstlal/Makefile \
	python/__init__.py \
	python/Makefile \
	python/pipeparts/Makefile \
	python/stats/Makefile \
	gst/Makefile \
	gst/debug/Makefile \
	gst/gst/Makefile \
	gst/lal/Makefile \
	gst/python/Makefile \
	bin/Makefile \
	share/Makefile \
	tests/Makefile \
	doc/Doxyfile \
	doc/Makefile \
	doc/gobject/Makefile
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
# Set the library API info
#
#  0.0.0   Original version
#  1.0.0   API used in 1.0.0


AC_SUBST([LIBAPI], [1])
AC_SUBST([LIBREL], [0])
AC_SUBST([LIBAGE], [0])
AC_SUBST([LIBVERSION], [${LIBAPI}:${LIBREL}:${LIBAGE}])


#
# Check for programs
#

# check for C99 compiler and C preprocessor
m4_pattern_allow([AC_PROG_CC_C99])
m4_ifdef([AC_PROG_CC_C99],[AC_PROG_CC_C99],[LALSUITE_AC_PROG_CC_C99])
AC_PROG_CPP

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
PKG_PROG_PKG_CONFIG(0.18.0)

MIN_DOXYGEN_VERSION="1.8.3"
GSTLAL_DOXYGEN([$MIN_DOXYGEN_VERSION])
AM_CONDITIONAL([COND_DOXYGEN], [test "x$HAVE_DOXYGEN" == "xyes"])
AC_SUBST([MIN_DOXYGEN_VERSION])

MIN_GTK_DOC_VERSION="1.11"
GTK_DOC_CHECK(["$MIN_GTK_DOC_VERSION"], [--flavour no-tmpl])
AC_SUBST([MIN_GTK_DOC_VERSION])


#
# Check for Python
#


AC_SUBST([MIN_PYTHON_VERSION], ["2.7"])
AM_PATH_PYTHON(["$MIN_PYTHON_VERSION"],,)
AX_PYTHON_DEVEL()
# hack to remove default lib dirs from PYTHON_LIBS.  only tested on Debian
# and SL6.  FIXME:  find a way to do this properly
PYTHON_LIBS=[`echo "$PYTHON_LIBS" | sed -e "s?-L/usr/lib[^[:space:]]*??g"`]


#
# Check for Avahi and DBus Python modules
#
# FIXME:  make their abscence fatal by adding ", fatal" to macro argument
#


AX_PYTHON_MODULE(avahi)
AX_PYTHON_MODULE(dbus)


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
# Check for GStreamer
#


AC_SUBST([GSTREAMER_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_VERSION], [1.14.1])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-allocators-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-controller-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your default gstreamer plugin search path, you might need to add the directory

	$plugindir
	
to your GST_PLUGIN_PATH environment variable.])
AC_SUBST([GSTLAL_PLUGIN_LDFLAGS], ["-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\|gstlal_\|GstLAL_\|GSTLAL_\).*"])

gstreamer_VERSION=`$PKG_CONFIG --modversion gstreamer-${GSTREAMER_RELEASE}`
AX_COMPARE_VERSION([$gstreamer_VERSION], [ge], [1.6.0], [
	AC_DEFINE([HAVE_GSTREAMER_1_6], [1], [Defined if gstreamer >= 1.6.x])
])
AX_COMPARE_VERSION([$gstreamer_VERSION], [ge], [1.14.0], [
	GSTPLUGINPREFIX="gst"
], [
	GSTPLUGINPREFIX=""
])
AC_SUBST([GSTPLUGINPREFIX])


#
# Check for GStreamer Video Library
#


AC_SUBST([GSTREAMER_VIDEO_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_VIDEO_VERSION], [1.2.4])
PKG_CHECK_MODULES([gstreamer_video], [gstreamer-video-${GSTREAMER_VIDEO_RELEASE} >= ${MIN_GSTREAMER_VIDEO_VERSION}])
AC_SUBST([gstreamer_video_CFLAGS])
AC_SUBST([gstreamer_video_LIBS])


#
# Check for GStreamer Audio Library
#


AC_SUBST([GSTREAMER_AUDIO_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_AUDIO_VERSION], [1.2.4])
PKG_CHECK_MODULES([gstreamer_audio], [gstreamer-audio-${GSTREAMER_AUDIO_RELEASE} >= ${MIN_GSTREAMER_AUDIO_VERSION}])
AC_SUBST([gstreamer_audio_CFLAGS])
AC_SUBST([gstreamer_audio_LIBS])


#
# Check for GObject Introspection
#


AC_SUBST([MIN_GOBJECT_INTROSPECTION_VERSION], [1.30.0])
GOBJECT_INTROSPECTION_CHECK([${MIN_GOBJECT_INTROSPECTION_VERSION}])
AC_SUBST([GIR_VERSION], [${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}])
AC_SUBST([GIR_VERSION_], [${GIR_VERSION/./_}])
AS_IF([test "x$found_introspection" = "xyes"], [
	AC_SUBST([girdir], ["\${datadir}/gir-1.0"])
	AC_SUBST([typelibdir], ["\${libdir}/girepository-1.0"])
	AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your gobject introspection typelib search path, you might need to add the directory

	$typelibdir

to your GI_TYPELIB_PATH environment variable.])
])


#
# Check for pygobject and gmodule (for gstpythonplugin from gstreamer)
#


# FIXME:  we really require 3.28 for element support, but not yet available on reference platform
AC_SUBST([MIN_PYGOBJECT_VERSION], [3.22])
PKG_CHECK_MODULES([PYGOBJECT], [pygobject-3.0 gmodule-2.0])
AC_SUBST([PYGOBJECT_CFLAGS])
AC_SUBST([PYGOBJECT_LIBS])
#AM_CHECK_PYTHON_LIBS(, AC_MSG_ERROR([Python libs not found.]))

# on Debian the file is /usr/lib/x86_64-linux-gnu/libpython2.7.so
# on SL7 the file is /usr/lib64/libpython2.7.so
# on raspberry pi the file is /usr/lib/aarch64-linux-gnu/libpython2.7.so
AC_SUBST([PYTHON_ABI_FLAGS], [])
AC_SUBST([PYTHON_LIB_SUFFIX], [so])
if test -a "/usr/lib/x86_64-linux-gnu/libpython2.7.so" ; then
	AC_SUBST([PYTHON_LIB_LOC], [/usr/lib/x86_64-linux-gnu])
elif test -a "/usr/lib/aarch64-linux-gnu/libpython2.7.so" ; then
	AC_SUBST([PYTHON_LIB_LOC], [/usr/lib/aarch64-linux-gnu])
elif test -a "/usr/lib64/libpython2.7.so" ; then
	AC_SUBST([PYTHON_LIB_LOC], [/usr/lib64])
elif test -a "/opt/local/lib/libpython2.7.dylib" ; then
	AC_SUBST([PYTHON_LIB_LOC], [/opt/local/lib])
	AC_SUBST([PYTHON_LIB_SUFFIX], [dylib])
	# FIXME:  .pc file broken on macs, remove when fixed
	PYGOBJECT_LIBS="${PYGOBJECT_LIBS} -lgmodule-2.0"
else
	AC_MSG_ERROR([libpython2.7.so not found])
fi


#
# Check for NumPy
#


AC_SUBST([MIN_NUMPY_VERSION], [1.7.0])
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


AC_SUBST([MIN_LAL_VERSION], [6.19.0])
AC_SUBST([MIN_LALMETAIO_VERSION], [1.4.0])
AC_SUBST([MIN_LALSIMULATION_VERSION], [1.8.0])
AC_SUBST([MIN_LALBURST_VERSION], [1.5.0])
AC_SUBST([MIN_LALINSPIRAL_VERSION], [1.8.0])
PKG_CHECK_MODULES([LAL], [lal >= ${MIN_LAL_VERSION} lalmetaio >= ${MIN_LALMETAIO_VERSION} lalsupport lalsimulation >= ${MIN_LALSIMULATION_VERSION} lalburst >= ${MIN_LALBURST_VERSION} lalinspiral >= ${MIN_LALINSPIRAL_VERSION}])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])

old_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS ${LAL_CFLAGS}"
AC_CHECK_DECL([LAL_PTHREAD_LOCK], [HAVE_LAL_PTHREAD_LOCK=yes], [HAVE_LAL_PTHREAD_LOCK=no], [#include <lal/LALConfig.h>])
CFLAGS="$old_CFLAGS"

AS_IF([test "x$HAVE_LAL_PTHREAD_LOCK" != "xyes"], [
	AC_MSG_FAILURE([lal must be built with pthread locking enabled])
])

AC_DEFUN([AX_PYTHON_SYMBOL], [
	AC_MSG_CHECKING([python module: from $1 import $2])
	AS_IF([$PYTHON -c "from $1 import $2"], [
		AC_MSG_RESULT(success)
	], [
		AC_MSG_ERROR([fail: are swig bindings missing?])
	])
])

AX_PYTHON_MODULE([lal], fatal)
AX_PYTHON_SYMBOL([lal], [LIGOTimeGPS])


#
# Check for glue, et al.
#


AC_SUBST([MIN_GLUE_VERSION], [2.0.0])
AX_PYTHON_GLUE([$MIN_GLUE_VERSION])
AC_SUBST([MIN_LIGO_SEGMENTS_VERSION], [1.2.0])
AX_PYTHON_LIGO_SEGMENTS([$MIN_LIGO_SEGMENTS_VERSION])
AC_SUBST([MIN_LIGO_LW_VERSION], [1.5.3])
AX_PYTHON_LIGO_LW([$MIN_LIGO_LW_VERSION])


#
# Check for GSL
#


PKG_CHECK_MODULES([GSL], [gsl])
AC_SUBST([GSL_CFLAGS])
AC_SUBST([GSL_LIBS])


#
# Check for ORC
#


AC_SUBST([MIN_ORC_VERSION], [0.4.16])
PKG_CHECK_MODULES([ORC], [orc-0.4 >= ${MIN_ORC_VERSION}], , [
	AC_MSG_WARN([ORC not found;  using potentially slower code paths.])
	ORC_CFLAGS="-DDISABLE_ORC"
])
AC_SUBST([ORC_CFLAGS])
AC_SUBST([ORC_LIBS])


#
# Check for zlib
#


AX_CHECK_ZLIB()
if test x$HAVE_LIBZ = "xno" ; then
	AC_MSG_FAILURE([zlib is required])
fi


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
