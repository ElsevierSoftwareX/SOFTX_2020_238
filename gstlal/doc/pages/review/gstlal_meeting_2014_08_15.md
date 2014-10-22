\page gstlalmeeting20140815page Review Meeting August 8-15, 2014 at CITA

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

All gstreamer elements (add the code names by hand later)

\section minutes minutes

Face-to-face review meeting 9-15 August 2014
Present: Kipp Cannon, Jolien Creighton, B.S. Sathyaprakash

Minutes:

- General comments:
- At some time soon we should review gstreamer elements associated with
trigger generation. We would also need to review the results for "first
two years publications - the injection element".
	- Action: Review committee must talk to Chad Hanna about what exactly
        is that we need to review. It is not clear how the review committee
        will be able to vett those results without having completed the review
        of gstlal as a whole. The time scale for this is January.

- Nightly build on Red Hat has problems but Debian seems to be fine.
        - Action: The problem should be identified and fixed.
        - Action item for Unit tests: Include output plots of unit tests
        under review documentation.

- Waveforms: Review of waveform injections is important but should we
consider that part of the gstlal review or "Waveform" review group.
        - Action: Sathya to find out who is reviewing lal-simulation.
        In particular waveform folks should review SpinTaylorT4 after TaylorF2.

- The question of ODC and on-line vetoes was discussed. It seems lal-gate
can handle gaps in data using state vectors and so gstlal will, in principle,
be ready to make use of ODC vetoes

- Action: Jolien Creighton to focus on the delta function test.

- Action: Kipp/Chad -> Perform auto-correlation test.

- Action: Kipp/Chad -> Template bank test -> Do a bunch of injections and look at the distribution
of SNR/<SNR>.

Codes reviewed and comments

____________________________________gstlal_whiten__________________________

- Reviewed with no actions
____________________________________gstlal_togglecomplex___________________

- Reviewed with no actions
____________________________________gstlal_sumsquares______________________

- Reviewed with no actions
____________________________________gstlal_statevector__________________

- Reviewed with no actions. Compatibility between what is provided by
DQ and what is masked or interpreted by gstlal is important as otherwise
the pipeline might break.

- Action: Reviewers to interact with DQ team to make sure statevectors
are compatible.
____________________________________gstlal_simulation__________________

- There are a number of issues, such as extra padding for injection series,
simulation series, etc., that need to be sorted out. At the moment it is also
not possible to get all the injections from a frame file. Also why is a
nano-second taken out at the beginning and added at the end. Get lal to
do a conditional taper at the start of the waveform and run it through
a high-pass filter (need to check if this is really needed for BNS).
Also need to find out if this should be the responsibility of waveform
developers or it could be done outside waveform generation.

- We need to look at this code again.

- Action: Please prepare a figure and a document to say what exactly
is being done to help complete the review.

- Action: Kipp will clean up this code and get it ready for review
(Perhaps we could take one of the montly telecons to complete this review)

____________________________________gstlal_segmentsrc__________________

- Reviewed with actions: Fix hard coded width. There is also a
*fixme* issue on line 498 that must be looked at.

- Write an illustration to describe how start and stop times of
segments are hanelded and if logic covers all cases possible.

- Notes: Many changes were made to this code during the review. It
should be checked again and we should have a look at it again
at some point.

____________________________________gstlal_reblock__________________

- Reviewed with no actions

____________________________________gstlal_nxydump__________________

- Reviewed with no actions (already before the f2f meeting)

____________________________________gstlal_nofakedisconts__________________

- Reviewed with no actions (minor action: fprintf should be changed to gerr)
lalchecktimestamps does unit test but it needs to be recorded somewhere.

____________________________________gstlal_matrixmixer__________________

- Reviewed with no actions

____________________________________gstlal_gate__________________

- Reviewed with actions: Set caps seem to have 64 but it is not implemented
in sink nor is a function available for type casting 64 bits (line 1322).

- Also please write a unit test.

____________________________________gstlal_firbank__________________

- Reviewed with no actions: Corrected a bug on line 1113. Please attach
to the review page outputs of unit tests using points for the plot and
zoommed in version.

____________________________________gstlal_drop__________________

- Reviewed with actions: Write some unit tests and show them to reviewers;
otherwise we are done with this code.

____________________________________gstlal_cachesrc________________________

- Reviewed with actions: Provide link to lal_cachesrc and other similar
files on the status page

____________________________________gds_lvshmsrc__________________

- Reviewed with actions: Please add output of tests that were done at
the f2f meeting to the review page.

