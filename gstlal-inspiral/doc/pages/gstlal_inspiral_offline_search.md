\page gstlalinspiralofflinesearchpage Offline search documentation

[TOC]

\section Introduction Introduction

Please see \ref gstlalinspirallowlatencysearchpage for background information.

\section Preliminaries Preliminaries

_NOTE: ANALYSIS BEST SUPPORTED AT UWM._

Running elsewhere reuquires dynamic Condor slots and modifcations to the gstlal_reference_psd, gstlal_inspiral and gstlal_inspiral_inj submit files.  We are working to standardize this on the LDG.

- Start by making a directory where you will run the analysis, e.g.,:

		$ mkdir /home/channa/test

\section makefiles Get example makefiles tailored to your application

-# Get two makefiles to set up the analysis dag.  One defines standard rules that should not need to be modified by the user, the other is use-case specific.  The examples 
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.offline_analysis_rules>Makefile.offline_analysis_rules</a> This makefile is required by all analysis
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.non_spinning_BNS>Makefile.non_spinning_BNS:</a>  Suitable for the 2014 BNS MDC with a nonspinning bank.
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.spinning_BNS>Makefile.spinning_BNS:</a>  Suitable for the 2014 BNS MDC with a spinning bank.
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.non_spinning_NSBH>Makefile.non_spinning_NSBH:</a>  Suitable for the 2014 NSBH MDC with a nonspinning bank.
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.spinning_NSBH>Makefile.spinning_NSBH:</a>  Suitable for the 2014 NSBH MDC with a spinning bank.
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.spinning_BBH>Makefile.spinning_BBH:</a>  Suitable for the 2014 BBH MDC with a spinning bank.
-# put the Makefiles in the analysis directory you made. 
-# Modify the Makefile to suit your analysis
-# run, e.g.,

		$ make -f Makefile.spinning_NSBH

-# Condor submit the resulting file

		$ condor_submit_dag trigger_pipe.dag

\section programs Programs used

- \ref gstlal_bank_splitter
- \ref gstlal_inspiral_pipe
- gstlal_cache_to_segments
- gstlal_segments_operations
- gstlal_segments_trim

