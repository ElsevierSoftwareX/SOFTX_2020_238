AC_DEFUN([AX_GSTLAL_SPLIT_VERSION], [
dnl save VERSION and AX_*_VERSION variables
_orig_VERSION=$VERSION
for part in MAJOR MINOR POINT ; do
	eval _orig_AX_${part}_VERSION=\$AX_${part}_VERSION
done
dnl split the argument
eval VERSION=\$$1
AX_SPLIT_VERSION
for part in MAJOR MINOR POINT ; do
	eval $1_${part}=\$AX_${part}_VERSION
done
dnl restore VERSION and AX_*_VERSION to orignal values
VERSION=$_orig_VERSION
unset _orig_VERSION
for part in MAJOR MINOR POINT ; do
	eval AX_${part}_VERSION=\$_orig_AX_${part}_VERSION
	unset _orig_AX_${part}_VERSION
done
])
