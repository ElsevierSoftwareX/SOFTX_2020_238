#
# Preamble
#


AC_INIT([gstlal],[0.1.0],[kcannon@ligo.caltech.edu],[gstlal])
AC_COPYRIGHT([Copyright (C) 2008 Kipp C. Cannon])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_FILES([ \
	Makefile \
	src/Makefile \
	src/plugins/Makefile \
	src/utilities/Makefile \
])


#
# Documentation boilerplate
#


AC_SUBST([docdir], [${datadir}/doc])
AC_SUBST([pkgdocdir], [${docdir}/${PACKAGE_NAME}-${PACKAGE_VERSION}])


#
# Check for programs
#


AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG()


#
# CFLAGS
#


AC_SUBST([AM_CFLAGS], [-Wall -Wextra -pedantic])


#
# Check for GStreamer 0.10
#


AC_SUBST([GSTREAMER_RELEASE], [0.10])
AC_SUBST([MIN_GSTREAMER_VERSION], [0.10.0])
PKG_CHECK_MODULES([gstreamer], [gstreamer-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION} gstreamer-base-${GSTREAMER_RELEASE} >= ${MIN_GSTREAMER_VERSION}], , [echo "Not Found!" ; exit 1])
AC_SUBST([gstreamer_CFLAGS])
AC_SUBST([gstreamer_LIBS])
AC_SUBST([plugindir], [${libdir}/gstreamer-${GSTREAMER_RELEASE}])
AC_SUBST([GSTREAMER_PLUGIN_LDFLAGS], ["-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*"])


#
# Check for LAL
#


PKG_CHECK_MODULES([LAL], [lal lalframe lalmetaio lalsupport], , [echo "Not Found!" ; exit 1])
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
