\page gstlalinspiralofflinesearchpage Offline search documentation

[TOC]

\section Introduction Introduction

Please see \ref gstlalinspirallowlatencysearchpage for background information.

\section Preliminaries Preliminaries

_NOTE: ANALYSIS BEST SUPPORTED AT UWM_

- start by making a directory where you will run the analysis, e.g.,:

		$ mkdir /home/channa/test

- get two makefiles to set up the analysis dag.  One defines standard rules that should not need to be modified, the other is use-case specific.

 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.offline_analysis_rules>Makefile.offline_analysis_rules</a>
 
 -# <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.triggers_example>Makefile.triggers_example</a>

The makefile will execute the following workflow (FIXME add program names to the boxes and urls)

@dotfile Makefile_offline_triggers.dot


