AC_DEFUN([AX_PYTHON_GLUE],[
	if test -z $PYTHON ; then
		PYTHON="python"
	fi
	PYTHON_NAME=`basename $PYTHON`
	AX_PYTHON_MODULE([glue])
	if test "x$HAVE_PYMOD_GLUE" == "xyes" ; then
		AC_MSG_CHECKING(glue version)
		GLUE_VERSION=`$PYTHON -c "from glue import __version__ ; print '.'.join(__version__.strip().split('.'))"`
		AC_MSG_RESULT($GLUE_VERSION)
	fi
])

AC_DEFUN([AX_PYTHON_LIGO_SEGMENTS],[
	if test -z $PYTHON ; then
		PYTHON="python"
	fi
	PYTHON_NAME=`basename $PYTHON`
	AX_PYTHON_MODULE([ligo.segments])
	if test "x$HAVE_PYMOD_LIGO_SEGMENTS" == "xyes" ; then
		AC_MSG_CHECKING(ligo.segments version)
		LIGO_SEGMENTS_VERSION=`$PYTHON -c "from ligo.segments import __version__ ; print '.'.join(__version__.strip().split('.'))"`
		AC_MSG_RESULT($LIGO_SEGMENTS_VERSION)
	fi
])
