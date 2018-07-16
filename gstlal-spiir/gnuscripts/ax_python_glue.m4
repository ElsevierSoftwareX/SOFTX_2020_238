AC_DEFUN([AX_PYTHON_GLUE],[
	if test -z $PYTHON ; then
		PYTHON="python"
	fi
	PYTHON_NAME=`basename $PYTHON`
	AX_PYTHON_MODULE([glue])
	if test "x$HAVE_PYMOD_GLUE" == "xyes" ; then
		AC_MSG_CHECKING(glue version)
		GLUE_VERSION=`$PYTHON -c "from glue import git_version ; print '%s.%s' % tuple((git_version.tag or '0-0').replace('glue-release-', '').split('-')[[:2]])"`
		if test "x${GLUE_VERSION}" == "x0.0" ; then
			GLUE_VERSION=
			AC_MSG_RESULT([unknown])
		else
			AC_MSG_RESULT($GLUE_VERSION)
		fi
	fi
])
