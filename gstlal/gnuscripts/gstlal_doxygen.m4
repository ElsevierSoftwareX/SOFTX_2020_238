AC_DEFUN([GSTLAL_DOXYGEN], [
	HAVE_DOXYGEN="no"
	AC_ARG_WITH([doxygen], [
		AS_HELP_STRING([--with-doxygen], [include doxygen documentation @<:@default=check@:>@])
	], [
	], [
		with_doxygen=check
	])
	AS_IF([test "x$with_doxygen" != "xno"], [
		AS_IF([test -x "$with_doxygen"], [
			DOXYGEN=$with_doxygen
		], [
			AC_PATH_PROG(DOXYGEN, doxygen)
		])
		AS_IF([test -x "$DOXYGEN"], [
			AC_MSG_CHECKING([doxygen is at least version $1])
			DOXYGEN_VERSION=`${DOXYGEN} --version`
			AX_COMPARE_VERSION(["$DOXYGEN_VERSION"], [ge], ["$1"], [
				AC_MSG_RESULT([yes])
				HAVE_DOXYGEN="yes"
			], [
				AC_MSG_RESULT([no ($DOXYGEN_VERSION)])
				AC_MSG_FAILURE([doxygen version: require >= $1])
			])
		], [
			AC_MSG_FAILURE([doxygen executable not found])
		])
	])
])
