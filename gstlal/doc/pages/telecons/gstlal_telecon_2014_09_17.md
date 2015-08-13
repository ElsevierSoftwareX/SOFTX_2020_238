\page gstlaltelecons20140917page Telecon Sept 17, 2014

\ref gstlalteleconspage

[TOC]

\section agenda Agenda

-# <a href=https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/>S6VSR3Replay</a>
-# Optimization efforts
    -# <a href=https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Optimization/O1/GSTLALAllSkyOffline3x3,Mt50/> Optimisation wiki entry for BBH </a>


\section attendance Attendance

Tjonnie, Steve P, Les, Chad, Jolien, FIXME.

\section minutes Minutes

Optimization efforts: gstlal_inspiral_flopulater and function call graphs

About gstlal_inspiral_flopulator:

 - assumes "time domain" filtering, and that all physical SNR time series are constructed
 - MFLOPS = 10^6 floating point operations per second 
 - possible effects on MFLOPS: 
    - SVD decomposition (across detectors)
    - time slice decomposition, template duration (across different sub-banks)

1) Les, analyzing 50Msun to 350Msun, 100 templates per bank, using non-optimized libraries

flopulater results: https://ldas-jobs.phys.uwm.edu/~lwade/Search/optimization_tests/IMBHB_templates/flopulate.txt

 - Chad: MFLOP per template is reasonable, but would like to see better SVD compression
   - should look into better lining up IMR filters

function call profiling: https://ldas-jobs.phys.uwm.edu/~lwade/Search/optimization_tests/IMBHB_templates/profile.txt


2) Tjonnie, running with 200 templates per sub bank, lower mass range, down to mchirp = 5Msun, and non-optimized libraries

flopulator: https://ldas-jobs.phys.uwm.edu/~tgfli/allflops.txt

 - ~ 2MFLOPS per template, consistent with higher sampling rates required for lower mass templates

function call profiling: https://ldas-jobs.phys.uwm.edu/~tgfli/profile.txt

In both cases, audioresample dominates the computational cost. BLAS is another high contender for computational cost, but subject to improvement after linking against optimized libraries installed at UWM.

Code issues:

 - Bugs in ER5 release have required patching and manual building. 
 - Les and Tjonnie have experienced trouble with compilation due to frame-cpp updates. 
 - All would like to see point releases deployed to the clusters as bugs are found in the future. 
 - These point releases should be recompiled whenever the underlying libraries are changed.
