\page gstlalmeeting20140305page Meeting on February 19, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- walk through gstlal_whiten.c 

\section minutes Minutes

- walked through some parts of gstlal_whiten.c focussing on data whitening and the part that applies the Hann window and handles overlapping segments.

Actions
- Kipp or Chad to create a separate gstlal_debug directory to host test codes and packges.
- Jolien to add his test package on lal_checktimestamps
- Fabien, Duncan and Sathya to install the test package 
- Kipp should send instructions/pipeline to test gstlal_whiten
- Review should at some point go through XLALPSDRegressor codes (there quite a few but in this category that have been written solely for gstlal).

Suggestions
- Jolien suggested to test gstlal_whiten using one noisy segment but with slighly different starting points to see if the code produces identical whitened data

Notes
- Chad and Kipp identified a bug in lal_shift and this has been now fixed (see the review call on Feb 26, 2014). Upon pressing Ctrl-C for the n-th time the code reports "shifting by n ns". This is misleading as the code really shifts for each Ctrl-C by only 1 ns. Please fix this.
- Kipp has been working on a unit test code for gstlal_matrixmixer but this is not ready yet but we will get back to checking gstlal_mixer once this code is completed.
- Sathya suggested that Chad and Kipp should come up with a more concrete agenda as to what the reviewers could do offline to accelerate the review. We are not at a stage where we (i.e. the reviewrs) are in a position to steer the direction of the review. Until such time Chad and Kipp should feel free to suggest assignments for the review team.
