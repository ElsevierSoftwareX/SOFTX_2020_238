#!/bin/sh

# Check with the user
echo "This script (re-)builds the source tree's configuration and build system."
read -p "Press CTRL-C to abort, or RETURN to continue... "

# Get it done
{
	{ echo "running aclocal (please ignore \"underquoted\" warnings)..." ; aclocal ; } &&
	#{ echo "running autoheader..." ; autoheader ; } &&
	{ echo "running automake..." ; automake -a -c ; } &&
	{ echo "running autoconf..." ; autoconf ; } &&
	echo "$0 complete." ;
} || { echo "$0 failed." ; false ; }
