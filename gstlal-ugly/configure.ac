#
# Preamble
#


AC_INIT([gstlal-ugly],[1.7.0],[gstlal-discuss@ligo.org],[gstlal-ugly])
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
	gstlal-ugly.spec \
	debian/control \
	debian/Makefile \
	lib/gstlal-ugly.pc \
	lib/Makefile \
	python/Makefile \
	share/Makefile \
	share/vis/Makefile \
	gst/Makefile \
	gst/nds/Makefile \
	gst/lal/Makefile \
	gst/python/Makefile \
	gst/framecpp/Makefile \
	gst/gds/Makefile \
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
AC_SUBST([MIN_PKG_CONFIG_VERSION], [0.18.0])
PKG_PROG_PKG_CONFIG("$MIN_PKG_CONFIG_VERSION")

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


AC_SUBST([MIN_PYTHON_VERSION], ["3.6"])
AM_PATH_PYTHON(["$MIN_PYTHON_VERSION"],,)
AX_PYTHON_DEVEL()
# hack to remove default lib dirs from PYTHON_LIBS.  only tested on Debian
# and SL6.  FIXME:  find a way to do this properly
PYTHON_LIBS=[`echo "$PYTHON_LIBS" | sed -e "s?-L/usr/lib[^[:space:]]*??g"`]
# hack to add missing options to PYTHON_LIBS.  sigh, what's the point of
# AX_PYTHON_DEVEL!?
PYTHON_LIBS="-module -avoid-version $PYTHON_LIBS"
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


AC_SUBST([MIN_NDS_VERSION], [0.11.5])
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


AC_SUBST([GSTREAMER_RELEASE], [1.0])
AC_SUBST([MIN_GSTREAMER_VERSION], [1.14.1])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}])
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
# Check for framecpp
#


AC_SUBST([MIN_FRAMECPP_VERSION], [2.6.2])
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
	framecpp_CXXFLAGS="${framecpp_CFLAGS} -D__STDC_CONSTANT_MACROS"
])
AC_SUBST([framecpp_CFLAGS])
AC_SUBST([framecpp_CXXFLAGS])
AC_SUBST([framecpp_LIBS])
AC_SUBST([framecpp_VERSION])

AS_IF([test "x$HAVE_FRAMECPP" != "xyes"], [AC_MSG_ERROR([framecpp not found])])


#
# Check for GDS
#


AC_SUBST([MIN_GDS_VERSION], [2.18.0])
AC_ARG_WITH(
	[gds],
	[AS_HELP_STRING([--with-gds], [include gds plugin @<:@default=check@:>@])],
	[],
	[with_gds=check]
)
AS_IF([test "x$with_gds" != "xno"], [
	PKG_CHECK_MODULES([gds], [gds >= ${MIN_GDS_VERSION}], [
		HAVE_GDS="yes"
		# hack to work around broken .pc file
		gds_LIBS="$gds_LIBS -llsmp"

		PKG_CHECK_MODULES([gds_framexmit], [gds-framexmit >= ${MIN_GDS_VERSION}], [
			HAVE_GDS_FRAMEXMIT="yes"
		], [
			AC_MSG_WARN([not found: $gds_framexmit_PKG_ERRORS])
			HAVE_GDS_FRAMEXMIT="no"
			# this component is required
			AC_MSG_WARN([gds install incomplete])
			HAVE_GDS="no"
		])
	], [
		AC_MSG_WARN([Not found: $gds_PKG_ERRORS])
		HAVE_GDS="no"
	])
], [
	AC_MSG_WARN([check for gds disabled])
	HAVE_GDS="no"
])
AM_CONDITIONAL([COND_GDS], [test "x${HAVE_GDS}" == "xyes"])
AS_IF([test "x${HAVE_GDS}" == "xyes"], [
	AC_DEFINE([HAVE_GDS], [1], [Defined if you have gds])  # a little penicillin will clear that right up
])
AC_SUBST([gds_CFLAGS])
AC_SUBST([gds_LIBS])
AC_SUBST([gds_framexmit_CFLAGS])
AC_SUBST([gds_framexmit_LIBS])


#
# Check for LAL
#


AC_SUBST([MIN_LAL_VERSION], [6.19.0])
AC_SUBST([MIN_LALMETAIO_VERSION], [1.4.0])
PKG_CHECK_MODULES([LAL], [lal >= ${MIN_LAL_VERSION} lalmetaio >= ${MIN_LALMETAIO_VERSION} lalsupport])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])

AX_PYTHON_MODULE([lal], fatal)


#
# Check for glue, et al.
#


AC_SUBST([MIN_GLUE_VERSION], [2.0.0])
AX_PYTHON_GLUE([$MIN_GLUE_VERSION])
AC_SUBST([MIN_LIGO_SEGMENTS_VERSION], [1.2.0])
AX_PYTHON_LIGO_SEGMENTS([$MIN_LIGO_SEGMENTS_VERSION])
AC_SUBST([MIN_LIGO_LW_VERSION], [1.5.3])
AX_PYTHON_LIGO_LW([$MIN_LIGO_LW_VERSION])
AC_SUBST([MIN_LIGO_SCALD_VERSION], [0.7.2])


#
# Check for gstlal
#


AC_SUBST([MIN_GSTLAL_VERSION], [1.6.0])
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


AC_SUBST([MIN_ORC_VERSION], [0.4.16])
PKG_CHECK_MODULES([ORC], [orc-0.4 >= ${MIN_ORC_VERSION}], , [
	AC_MSG_WARN([ORC not found;  using potentially slower code paths.])
	ORC_CFLAGS="-DDISABLE_ORC"
])
AC_SUBST([ORC_CFLAGS])
AC_SUBST([ORC_LIBS])


#
# Check for webdir, this enables installation of web visualation tools
#


AC_ARG_WITH(
	[webdir],
	[AS_HELP_STRING([--with-webdir], [specify location of directory for web visualization tools. [$HOME/public_html]])],
	[],
	[with_webdir=no]
)
AS_IF([test "x$with_webdir" != "xno"], [
	AC_MSG_WARN([copying web visualization tools to $with_webdir])
	AC_SUBST([WEBDEST],["$with_webdir"])
	HAVE_WEBDIR="yes"
],[
	AC_MSG_WARN([disabling installation of web visualization tools])
	HAVE_WEBDIR="no"
])
AC_DEFINE([HAVE_WEBDIR], [test "x{HAVE_WEBDIR}" == "xyes"], [Defined if you are enabling web visualiation])
AM_CONDITIONAL([COND_WEBDIR], [test "x${HAVE_WEBDIR}" == "xyes"])


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
