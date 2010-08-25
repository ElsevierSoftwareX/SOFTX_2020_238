#
# Preamble
#


AC_INIT([gstlal],[0.1.0],[kipp.cannon@ligo.org],[gstlal])
AC_COPYRIGHT([Copyright (C) 2008 Kipp C. Cannon])
# a file whose existance can be used to use to check that we are in the
# top-level directory of the source tree
AC_CONFIG_SRCDIR([gnuscripts/ltmain.sh])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_FILES([ \
	Makefile \
	src/data/Makefile \
	src/data/www/Makefile \
	src/data/www/assets/Makefile \
	src/data/www/assets/fancybox/Makefile \
	src/Makefile \
	src/bin/Makefile \
	src/lib/gstlal.pc \
	src/lib/Makefile \
	src/skymap/Makefile \
	src/cairovis/Makefile \
	src/plugins/Makefile \
	src/plugins/python/Makefile \
	src/python/Makefile \
	src/python/elements/Makefile \
	src/utilities/Makefile \
	tests/Makefile
])


#
# Extra directories
#


AC_SUBST([docdir], [${datadir}/doc])
AC_SUBST([pkgdocdir], [${docdir}/${PACKAGE_NAME}-${PACKAGE_VERSION}])
AC_SUBST([pkgconfigdir],[${libdir}/pkgconfig])


#
# Check for programs
#


AC_PROG_CC_STDC
AC_PROG_INSTALL
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG()

#
# Check for Python
#


AM_PATH_PYTHON([2.4],,)
AX_PYTHON_DEVEL()


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
# Check for NDS2
#


AC_ARG_WITH(
	[nds],
	[AS_HELP_STRING([--with-nds], [include NDS source element @<:@default=check@:>@])],
	[],
	[with_nds=check]
)
AS_IF(
	[test "x$with_nds" != "xno"],
	[PKG_CHECK_MODULES(
		[NDS],
		[nds2-client],
		[AC_DEFINE([HAVE_NDS], [1], [Define if you have nds2-client])
		HAVE_NDS="yes"],
		[HAVE_NDS="no"
		AS_IF(
			[test "x$with_nds" != "xcheck"],
			[AC_MSG_ERROR([Not found!])],
			[AC_MSG_WARN([Not found!])]
		)]
	)]
)
AM_CONDITIONAL([COND_NDS], [test "x${HAVE_NDS}" == "xyes"])
AC_SUBST([NDS_CFLAGS])
AC_SUBST([NDS_LIBS])


#
# Check for cairo
#


AC_ARG_WITH(
	[cairo],
	[AS_HELP_STRING([--with-cairo], [include cairo visualization elements @<:@default=check@:>@])],
	[],
	[with_cairo=check]
)
AS_IF(
	[test "x$with_cairo" != "xno"],
	[PKG_CHECK_MODULES(
		[cairo],
		[cairo],
		[AC_DEFINE([HAVE_CAIRO], [1], [Define if you have cairo])
		HAVE_CAIRO="yes"],
		[HAVE_CAIRO="no"
		AS_IF(
			[test "x$with_cairo" != "xcheck"],
			[AC_MSG_ERROR([Not found!])],
			[AC_MSG_WARN([Not found!])]
		)]
	)]
)
AM_CONDITIONAL([COND_CAIRO], [test "x${HAVE_CAIRO}" == "xyes"])
AC_SUBST([cairo_CFLAGS])
AC_SUBST([cairo_LIBS])


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
# Check for PyGObject
#
PKG_CHECK_MODULES([pygobject], [pygobject-2.0], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([pygobject_CFLAGS])
AC_SUBST([pygobject_LIBS])


#
# Check for liboil 0.3
#


AC_SUBST([LIBOIL_RELEASE], [0.3])
AC_SUBST([MIN_LIBOIL_VERSION], [0.3.14])
PKG_CHECK_MODULES([liboil], [liboil-${LIBOIL_RELEASE} >= ${MIN_LIBOIL_VERSION}], , [AC_MSG_ERROR([Not found!])])
AC_SUBST([liboil_CFLAGS])
AC_SUBST([liboil_LIBS])

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


PKG_CHECK_MODULES([LAL], [lal lalframe lalmetaio lalsupport lalburst lalinspiral], , [AC_MSG_ERROR([Not found!])])
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
