AC_DEFUN([AX_PYTHON_PYLAL],[
	if test -z $PYTHON ; then
		PYTHON="python"
	fi
	PYTHON_NAME=`basename $PYTHON`
	AX_PYTHON_MODULE([pylal])
	if test "x$HAVE_PYMOD_PYLAL" == "xyes" ; then
		AC_MSG_CHECKING(pylal version)
		PYLAL_VERSION=`$PYTHON -c "from pylal import git_version ; print '%s.%s' % tuple((git_version.tag or '0-0').replace('pylal-', '').split('-')[[:2]])"`
		if test "x${PYLAL_VERSION}" == "x0.0" ; then
			PYLAL_VERSION=
			AC_MSG_RESULT([unknown])
		else
			AC_MSG_RESULT($PYLAL_VERSION)
		fi
	fi
])
