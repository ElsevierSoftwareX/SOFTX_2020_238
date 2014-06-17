\page gstlalinspiralofflinesearchpage Offline search documentation

[TOC]

\section Introduction Introduction

Please see \ref gstlalinspirallowlatencysearchpage for background information.

\section Preliminaries Preliminaries

_NOTE: ANALYSIS BEST SUPPORTED AT UWM._

Running elsewhere reuquires dynamic Condor slots and modifcations to the gstlal_reference_psd, gstlal_inspiral and gstlal_inspiral_inj submit files.  We are working to standardize this on the LDG.

- Start by making a directory where you will run the analysis, e.g.,:

		$ mkdir /home/channa/test

- Get two makefiles to set up the analysis dag.  One defines standard rules that should not need to be modified, the other is use-case specific.  The examples 

 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.offline_analysis_rules>Makefile.offline_analysis_rules</a>
 
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.triggers_example>Makefile.triggers_example</a>

will need to be modified to your situation.

The makefile will execute the following workflow that culminates in an HTCondor DAG

@dotfile Makefile_offline_triggers.dot

To see the HTCondor DAG please see the documenation for gstlal_inspiral_pipe

The DAG can be submitted via:

		$ condor_submit_dag trigger_pipe.dag

When the workflow is finished you should see a web page output in the directory specified in the Makefile
