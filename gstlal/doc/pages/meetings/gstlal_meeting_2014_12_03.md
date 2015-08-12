\page gstlalmeeting20141203page Review Meeting December 03, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

  - https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/python/inspiral.py
  - https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/python/streamthinca.py

We could only review the two codes above. The following items will be looked at the next (f2f) review.
  - https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/doc/gstlal-inspiral-0.4.1/html/gstlalinspiralautochisqlenstudypage.html
  - https://ldas-jobs.phys.uwm.edu/~gstlalcbc/MDC/


In attendance: Kipp Cannon, Chad Hanna, Jolien Creighton, Florent Robinet, B. Sathyaprakash, Duncan Meacher, T.G.G. Li

<!---
\section action Action Items

Action items on inspiral.py
  - Document examples of how to get SNR history, etc., to a web browser in an offline search
  - Long term goal: Using template duration (rather than chirp mass) should load balance the pipeline and improve statistics
  - L651: One thing to sort out is the signal probability while computing coincs
  - L640-L647: Get rid of obsolete comments 
  - L667: Make sure timeslide events are not sent to GRACEDB
  - Lxxx: Can normalisation of the tail of the distribution pre-computed using fake data?
  - L681: fmin should not be hard-coded to 10 Hz. horizon_distance will be horribly wrong if psd is constructed, e.g. using some high-pass filter. For example, change the default to 40 Hz.
  - L817: If gracedb upload failed then it should be possible to identify the failure, the specifics of the trigger that encountered failure and a way of submitting the trigger again to gracedb is important. Think about how to clean-up failures.
  - Mimick gracedb upload failures and see if the code crashes

Action items on streamthinca.py

 - Question: Is it possible for the offline pipeline to begin producing tiggers after a certain time rather than waiting for all the inspiral jobs to get over? Will be particularly useful if the data length is ~ months or ~ year. Should also avoid producing massive amount of data, right?

 - L300+: Please document within the code that the FAR column is used to store FAP so that future developers don't get confused what that column represents
-->

\section minutes minutes

1. inspiral.py: 
  - L379: In principal it will be possible to use inspiral.py to do a number of time slides by giving a pre-specified time slides table.

  - It will be possible to change SNR threshold 

  - L536: thinca_interval can be reduced to improve latency; changing it won't change the result but increasing it will increase latency. Chad has used an interval of 1 s  (triggers are produced every second) to get lowest latency.

2. snglcoinc.py and ligolw_thinca.py

No comments:

3. streamthinca.py
