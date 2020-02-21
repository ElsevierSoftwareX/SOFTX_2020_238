#
# AX_PYTHON_GLUE([MINVERSION = 0])
#
AC_DEFUN([AX_PYTHON_GLUE],[
	AC_REQUIRE([AM_PATH_PYTHON])
	AX_PYTHON_MODULE([glue])
	AS_IF([test "x$HAVE_PYMOD_GLUE" == "xyes"], [
		AC_MSG_CHECKING(glue version)
		GLUE_VERSION=`$PYTHON -c "from glue import __version__ ; print('.'.join(__version__.strip().split('.')))"`
		AS_IF([test $? != "0"], [
			AC_MSG_ERROR(["cannot determine version"])
		])
		minversion=$1
		AX_COMPARE_VERSION([$GLUE_VERSION], [ge], [${minversion:-0}], [
			AC_MSG_RESULT([$GLUE_VERSION])
		], [
			AC_MSG_WARN([found $GLUE_VERSION, require at least $1])
		])
		unset minversion
	])
])

#
# AX_PYTHON_LIGO_SEGMENTS([MINVERSION = 0])
#
AC_DEFUN([AX_PYTHON_LIGO_SEGMENTS],[
	AC_REQUIRE([AM_PATH_PYTHON])
	AX_PYTHON_MODULE([ligo.segments])
	AS_IF([test "x$HAVE_PYMOD_LIGO_SEGMENTS" == "xyes"], [
		AC_MSG_CHECKING(ligo.segments version)
		LIGO_SEGMENTS_VERSION=`$PYTHON -c "from ligo.segments import __version__ ; print('.'.join(__version__.strip().split('.')))"`
		AS_IF([test $? != "0"], [
			AC_MSG_ERROR(["cannot determine version"])
		])
		minversion=$1
		AX_COMPARE_VERSION([$LIGO_SEGMENTS_VERSION], [ge], [${minversion:-0}], [
			AC_MSG_RESULT([$LIGO_SEGMENTS_VERSION])
		], [
			AC_MSG_WARN([found $LIGO_SEGMENTS_VERSION, require at least $1])
		])
		unset minversion
	])
])
