#!/bin/sh

# Check with the user
echo "This script (re-)builds the source tree's configuration and build system."
read -p "Press CTRL-C to abort, or RETURN to continue... " INPUT

# Get it done
{
	{ echo "running aclocal (please ignore \"underquoted\" warnings)..." ; aclocal -I gnuscripts ; } &&
	#{ echo "running libtoolize..." ; libtoolize -c -f || glibtoolize $LIBTOOLIZE_FLAGS ; } &&
	#{ echo "running autoheader..." ; autoheader ; } &&
	{ echo "running automake..." ; automake -a -c ; } &&
	{ echo "running autoconf..." ; autoconf ; } &&
	echo "$0 complete." ;
} || { echo "$0 failed." ; false ; }
