\page gstlalmeeting20140120page Meeting on January 20, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- Introductions, etc.  
- gstlal documentation:

\section minutes Minutes

Actions
- update links in project page (springboard): done.
- make an example page to show reviewers how edit documentation, etc.: Email sent

Chads Actions
- Investigate the use of make_whitened_multirate_src() vs the whitener in gstlal_fake_frames
- lal_shift: make a unit test
- Fix order of capsfilter / audioresample in gstlal_fake_frames graph Commit: b8d40a78d2484b32867ef06b7cc574871725f589
- Add dot graph output for gstlal_fake_frames Commit: b8d40a78d2484b32867ef06b7cc574871725f589
- make dot graph get dumped by env variable. Commit: 434a2d61eb5611817444309398d0859018dfed86

Notes:

- We might not need to review gstreamer elements
- Sathya wants to see the review lead to broader use of gstlal
- Kipp suggest that although we may have a specific task (e.g. review gracedb events as "the" final product) we will review a lot of stuff used for other purposes.
- Start by looking at fake data generation

