#!/bin/sh

# Check with the user
echo "This script (re-)builds the source tree's configuration and build system."
read -p "Press CTRL-C to abort, or RETURN to continue... " INPUT

# Get it done
{
	{ echo "running autoreconf" ; autoreconf ; } &&
	echo "$0 complete." ;
} || { echo "$0 failed." ; false ; }
