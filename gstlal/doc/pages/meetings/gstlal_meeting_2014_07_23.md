\page gstlalmeeting20140723page Review Meeting July 23, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- See completed action items here: \ref gstlalinspirallowlatencysearchpage 
- Kipp fixed the audioresample element: https://bugzilla.gnome.org/show_bug.cgi?id=732908
- Les presentation on IMBHB event ranking: https://dcc.ligo.org/LIGO-G1400796
- Cody presentation on FAP estimation normalization procedure: https://dcc.ligo.org/G1400798 
- S6 replay mdc \ref gstlalinspirals6replaypage 
- Unit tests:  https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal/tests
 - lal_firbank:  confirm that FIR filter is an identity transform when given a unit impulse response;  test at single- and double-precision, both time-domain and frequency-domain modes.
 - lal_matrixmixer:  confirm that matrix mixer is an identity transform when given an identity matrix;  test at single and double precision, one- and two-channel input.  also feed one single buffer into matrix mixer and compare output buffer to matrix-matrix multiply performed by numpy;  test at single and double precision.
 - audioresample:  checks audioresample element for timestamp drift (an old bug we fixed years ago).
 - lal_whiten:  confirm that whitener is an identity transform when given a white PSD, and histogram output of whitener when fed coloured noise to confirm it is a Gaussian with the correct width.  only identity test of automated.
 - other:
  - framesrc:  feeds a group of frame files into framecpp_channeldemux to see what happens.  hard-coded to my home directory, not yet useful as part of a general test suite
  - lal_checktimestamps:  uses lal_shift element to add 1 ns of offset to buffer timestamps on user input to stimulate a response from lal_checktimestamps.  cannot yet be used as part of an automated test suite.
  - lal_reblock
 - unit tests make use of "cmp_nxydumps.py" (https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal/tests/cmp_nxydumps.py) which contains the logic for "smart" (a.k.a., sloppy) comparison of time series data.

\section minutes minutes

