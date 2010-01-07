#
# Preamble
#


AC_INIT([gstlal],[0.1.0],[kipp.cannon@ligo.org],[gstlal])
AC_COPYRIGHT([Copyright (C) 2008 Kipp C. Cannon])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_FILES([ \
	Makefile \
	src/Makefile \
	src/lib/Makefile \
	src/plugins/Makefile \
	src/python/Makefile \
	src/python/elements/Makefile \
	src/utilities/Makefile \
	tests/Makefile
])


#
# Documentation boilerplate
#


AC_SUBST([docdir], [${datadir}/doc])
AC_SUBST([pkgdocdir], [${docdir}/${PACKAGE_NAME}-${PACKAGE_VERSION}])


#
# Check for programs
#


AC_PROG_CC_STDC
AC_PROG_INSTALL
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG()
AM_PATH_PYTHON([2.4],,)


#
# CFLAGS
#


AX_CFLAGS_WARN_ALL([AM_CFLAGS])
AM_CFLAGS="$AM_CFLAGS -Wextra -Wno-missing-field-initializers -Wno-unused-parameter"	# extra gcc-specific stuff
AC_SUBST([AM_CFLAGS])
AX_CC_MAXOPT


#
# Check for math library
#


AC_CHECK_LIB([m], [main], , [AC_MSG_ERROR([Not found!])])


#
# Check for FFTW
#


PKG_CHECK_MODULES([FFTW], [fftw3], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([FFTW_CFLAGS])
AC_SUBST([FFTW_LIBS])


#
# Check for GStreamer 0.10
#


AC_SUBST([GSTREAMER_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_VERSION], [0.10.0])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_MSG_NOTICE([If you have chosen to install the software in a location not included in your gstreamer's default search path, you might need to add the directory

	$plugindir
	
to your GST_PLUGIN_PATH environment variable.])
AC_SUBST([GSTREAMER_PLUGIN_LDFLAGS], ["-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*"])


#
# Check for liboil 0.3
#


AC_SUBST([LIBOIL_RELEASE], [0.3])
AC_SUBST([MIN_LIBOIL_VERSION], [0.3.14])
PKG_CHECK_MODULES([liboil], [liboil-${LIBOIL_RELEASE} >= ${MIN_LIBOIL_VERSION}], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([liboil_CFLAGS])
AC_SUBST([liboil_LIBS])


#
# Check for LAL
#


PKG_CHECK_MODULES([LAL], [lal lalframe lalmetaio lalsupport lalburst], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([LAL_CFLAGS])
AC_SUBST([LAL_LIBS])


#
# Check for GSL
#


AX_PATH_GSL()


#
# Output configure information
#


AC_PROG_MAKE_SET
AC_OUTPUT
