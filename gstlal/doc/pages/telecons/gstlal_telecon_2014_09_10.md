\page gstlaltelecons20140910page Telecon Sept 10, 2014

\ref gstlalteleconspage

[TOC]

\section agenda Agenda

-# Code status
  -# Ranking statistic work on master
  -# New gstlal-burst package
-# Status of audioresample bug fix and lscsoft
-# Any other business
  -# [S6VSR3Replay](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/)
  -# [Optimisation wiki entry (BBH)](https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Optimization/O1/GSTLALAllSkyOffline3x3,Mt50)


\section attendance Attendance

Chad H., Chris P., Duncan M., Kipp C., Les W., Patrick B., Ryan E., Tjonnie L., Sarah C., Laleh, Jolien C., Cody M., Sathya, Ian H.


\section minutes Minutes

 * showed results of Gaussian BNS MDC with new ranking statistic code
 * Tjonnie will take a stab and automating the tuning of the numerator for different mass ranges
 * gstlal-burst package will be ready for a release in a couple of weeks.  the rest will probably not be ready so we'll do a point release of gstlal-ugly with the burst files removed to allow gstlal-ugly to move through the repos and get installed
 * audioresample bug.  will be putting together a patched 0.10.36 for Debian and seeing if it's possible to build a complete set of 0.10.36 packages (with the audioersample patch) on CIT
 * discussed S6 replay, relationship with gstlal inspiral review, relationship with pycbc readiness.
  * need to identify exactly what data will be analyzed so that gstlal inspiraland pycbc results can be compared, especially so that their reviews can meaningfully conclude something about their equivalence
  * although there is still the offline ihope run from S6 that can be used for comparison with gstlal
  * an offline analysis with gstlal should be sufficient for a review.  can be completed in advance of the replay (which will be throttled to run in sync with wall clock time)
