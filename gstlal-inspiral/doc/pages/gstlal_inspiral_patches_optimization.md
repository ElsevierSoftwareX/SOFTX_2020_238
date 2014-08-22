\page gstlalinspiralpatchinformation gstlal inspiral code optimization patches

\section Introduction
This page documents patches used in Makefile.ligosoftware, which installs lalsuite and gstlal with optimized dependencies on a vanilla machine.  Information on the Makefile can be found on the \ref gstlalinspiralcodeoptimization page.  The patches themselves, as well as the Makefile, can be found at

https://ldas-jobs.ligo.caltech.edu/~cody.messick/vanilla_make/

\section gstlal gstlal patches
-  datasource.py
Patch adds an option to choose a dataquality channel name other than the default

-  gstlal_peakfinder.c & gstlal_peakfinder.ct
Patches fixes improve peakfinding efficiency

\section gstlal-inspiral gstlal-inspiral patches
-  gstlal_inspiral
Patch makes online analysis use a random port number for website updating during analysis run to avoid the potential problem of multiple jobs on a single node trying to use the same port

-  servicediscovery.py
Patch makes online analysis not use avahi, which at the time of release was not being used on the clusters yet and had not been configured correctly in the code
