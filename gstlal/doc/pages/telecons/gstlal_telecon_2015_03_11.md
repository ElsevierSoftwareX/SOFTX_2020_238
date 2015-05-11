\page gstlaltelecons20150311page Telecon March 11, 2015

\ref gstlalteleconspage

[TOC]

\section agenda Agenda

 - UTCToGPS Bug Status (Cody)
   - PR: https://bugs.ligo.org/redmine/issues/1916
 - BBH Search

\section attendance Attendance
Cody M.
Kent
Kipp
Laleh
Patricia
Sarah
Surabhi
Tjonnie
Jolien
Stephen
Patrick

\section action Action Items
  - For Cody: Check that changes in compilers haven't caused problems. Check that a temporary variable was used when checking gps_time_now value. Do another sanity check that the times are actually valid. Try replacing code with code to return random struct with times that have failed in the past.
* For Jolien: Check the behavior of gstreamer and lal's raise function. Change siminspiralFDTD routine to take df.
  - For Kipp: Change swig code so it just copies values in SWIGPython.i
  - For discussion next time: Should likelihood ranking stat include triggers found in coincidence? Question based on rates/significance group discussion.

\section minutes Minutes


  - UTCToGPS Issues
    - Cody's summary
      - In early Feb., a bug arose and started randomly throwing an error saying not valid date/time structure. He's currently working with two commits to get things working. Not sure if it's related to Reinhard's commits. Could be memory error.
    - Kipp: The history is that Reinhard commited a patch which made pylal uncompilable. The short term fix was to find time conversion in gstlal and switch to pylal. But turntypes came out different. But, in any event, Cody is reporting a different problem unrelated to types.
    - Cody: This popped up before Reinhard's change anyway.
    - Kipp: 
      - gps_time_now = lal.UTCToGPS(time.gmtime()) gives error message of ValueError: in method 'UTCToGPS', argument 1 of type 'struct tm const * (invalid date/time)' Time struct is different in python and C; There are invalid time errors (ie. month 0 in time library). 
      - But Cody checked that conversion for this exists in lal.
    - Jolien: Conversion is in swig code.
    - Kipp: The type map is in swig language to convert struct objects from python to C.
    - Jolien: This is irrelevant because all these changes can be reproduced before swig changes.
    - Kent: Incompatible compilers?
    - Cody: Always compiled on UWM with optimized libraries.
    - Kipp: ICC and GCC mix
    - Kent: Reviewing compiler history/changes is probably a good place to start.
    - Kipp: Reinhard's commit probably had nothing to do with it. It just precipitated the change to swig bindings. Have people been running dags?
    - Kent: I've been running BBH dags with current gstlal and lalsuite.
    - Jolien: Chris Pankow's burst pipeline had trouble with appsink handler. It might be a related problem. The PR is here: https://bugs.ligo.org/redmine/issues/1964 The summary is that lal raises a segmentation violation but code continues along. Does gstreamer allow this? Would gstreamer override lal's raise function?
    - Kipp: It could. For example, XLALSetSilentHandler().
    - Jolien: I will check that this affects actual raise routine.
    - Cody: Another weird thing: when UTCtoGPS error came up, it didn't cause dag to fail originally; it does now.
    - Kent: Condor has evolved so you might have to go back a few commits to have reproducible environment.
    - Cody: Changed gps_time_now = lal.UTCToGPS(time.gmtime()) to get temporary variable with try/except and created workaround. But I will look at this again to make sure I am using a temporary variable
    - Jolien: Are the times actually valid?
    - Cody: It does appear that everything matched convention; I will do another sanity check.
    - Jolien: Since this started around Feb 10 can we set system clock to last December.
    - Kipp: No because we would have to do that on cluster.
    - Cody: I have also tried a stress test with UTCtoGPS.
    - Jolien: You could also try replacing code with code to return random struct with one's that have failed in the past.
    - Kipp: Code fails on line 201 of lal/swig/SWIGPython.i equivalent pylal code is here: https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/pylal/src/xlal/date.c#n52
   - Jolien: XLALFillBrokenDownTime defined here: https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/_x_l_a_l_civil_time_8c_source.html#l00347
    - Kipp: May be place where it's not threadsafe in swig bindings
    - Jolien: We could change swig code so it just copies values in SWIGPython.i
  - BBH Search
    - Tjonnie: We have been testing siminspiralFDTD routine. Some code requires rewriting since df is not a given variable any more
    - Jolien: We could probably just change it to take df instead of dt; I will do this
    - Stephen: Any updates on using ROM; trying to use siminspiralFD; epoch is set correctly; if we have ability to just set df then we are very close to using ROM
    - Stephen: Kent, time to followup on ER6 loud zerolag event? Why did it make it through pipline?
    - Kent: Not yet
    - Kent: I've been running on CIT and found lots of places where memory requirements need to be set. Also runtime is an issue. CIT is 4hr eviction time window. UWM has week time limit. Thoughts on lowering memory requirements? None offered
    - Jolien: Question for Kipp: Should likelihood ranking stat include triggers found in coincidence?
    - Kent: More data analyzed with ER6 runs on CIT; numbanks = 4; Chad suggested 10 and 15
    - Steve: Chad suggested more subbanks per job; right for background estimation; a single job has a single background
    - Jolien: Background distributions from last year webpages were lots; but now there are just one set of plots. Any one know why?
