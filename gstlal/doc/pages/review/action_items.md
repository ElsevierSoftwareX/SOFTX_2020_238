\page gstlalactionitemspage Action Items

\section action_items General Action Items 


- Write documentation for autochisq (paper in progress)
- Explore autocorrelation chisquared mismatch scaling with number of samples e.g., @f$ \nu + \epsilon(\nu) \delta^{2} @f$
- Make sure capsfilters and other similar elements are well documented within graphs (e.g. put in rates, etc)
- Add description of arrows in graphs
- Feature request for detchar - It would be helpful to have online instrument state that could be queried to know if an instrument will be down for an extended time
- figure out what the PE people want to present and when;  it should be
  related to this subject matter of this review meeting
- get documentation generated and installed
- explain why approximating transition from signals invisible to the next most sensitive instrument to certainly visible by convolving hard edge with \chi distribution with hard edge at detection threshold is a good idea for joint SNR PDFs
- Run the pipeline with Gaussian noise with the color expected in O1/O2/O3/aLIGO Design (no need to run on all, one or two will do) with BNS template waveforms with and without spin
- Write as many unit tests as possible

<!---
These elements had general action items which have been moved to their source code
	- whiten
	- lal_cachesrc
	- lal_drop
	- lal_adder
	- lal_firbank
	- lal_itac
-->


\section telecon2015_03_11 March 11, 2015 telecon
\ref gstlaltelecons20150311page

- For Cody: Check that changes in compilers haven't caused problems. Check that a temporary variable was used when checking gps_time_now value. Do another sanity check that the times are actually valid. Try replacing code with code to return random struct with times that have failed in the past.
* For Jolien: Check the behavior of gstreamer and lal's raise function. Change siminspiralFDTD routine to take df.
- For Kipp: Change swig code so it just copies values in SWIGPython.i
- For discussion next time: Should likelihood ranking stat include triggers found in coincidence? Question based on rates/significance group discussion.
	
\section meeting2015_01_12 January 12-18, 2015 meeting
\ref gstlalmeeting20150112page

- Compute the actual expected SNR (instead of from average PSD) and plot the SNR histograms again
- We should compare the on-line and off-line results and understand the similarities and differences
- Figure out why multiple low-SNR events get the same likelihood (these constitute about a quarter of all events)
- Review of Summary pages at:
<https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/MDC/BNS/Summer2014/recolored/nonspin/966384015-971384015-pipe-compare-CAT2/ALL_LLOID_COMBINED_openbox.html?ALL_LLOID_COMBINED_openbox_summary.html>
- Make zoomed versions for accuracy plots
- Make Histogram of accuracies
- Make accuracy plots as a function of FAR
- Plot accuracy as a function of SNR
- Injections are found with `pi` time-shift; probably coming from one time-slide (check and fix)
- What is the reason for close-by missed injections in missed-found plot (as a function of Mchirp)?
- Perpahs the prefactors_range (0.0, 0.10) might be too narrow
- Figure out the reason for the dip in the foreground plot (in SNR-Chisq section)
- Make efficiency plots (efficiency as a function of distance, effective distance and chirp distance)
- Compute the Range as a function of network for an IDEAL pipeline (T1200458, Figure 2 would be a good example)
- Update the online instructions including Makefile

- Review of GRACE-DB:
 - Check live times.
 - It would be useful to have a Table of CPU usage rather than for individual nodes.
 - Currently, on-line analysis requests a headroom of 20% CPU. Can this be defended? Explore running some `nice` jobs in the background and see if this affects performace
 - The particular event we saw had a 17 minute latency for producing sky-map: https://gracedb.ligo.org/events/view/T124866

\section telecon2015_01_21 January 21, 2015 telecon
\ref gstlaltelecons20150121page

- Chad will run BBH Makefile and converse with Tjonnie about stalling processes.
- Chad will  write a patch to adjust condor submit files to make it more vanilla so that if a cluster has dynamic slots, they will match.
- Les and Tjonnie will coordinate computational requirements calculations for BBH and IMBH searches


\section meeting2014_12_03 December 3, 2014 meeting
\ref gstlalmeeting20141203page

-Action items added to inspiral.py

-Action items added to streamthinca.py


\section meeting2014_11_05 November 5, 2014 meeting
\ref gstlalmeeting20141105page

- Chad: To amend the documentation to warn about fixed SNR threshold of 4.  At some point in the future consider making SNR threshold an adjustable parameter

- Kipp: Need to fix distance quantization for the construction of SNR PDFs

- Chad: You have a hard-coded 4096 Hz, find out where that is; need to have a variable sampling rate (perhaps this is not hardcoded any more???).

- Chad: Time-dependence of horizon distances can be captured in the computation of the 'numerator'

- Chad: Indicate that overlap should be 10% of the num_split_templates

- Chad: People shouldn't have to remove Multicore=True

- Chad:  The length of autocorrelation chi-squared in sample points is set to AC_LENGTH = 351. It might be worth exploring how the sensitivity volume changes by varying AC_LENGTH

- Chad: Please specify the boundaries of the various parameters in Makefile, especially if it is desirable to explore how the sensitivity changes for different values of these parameters

- Chad: Makefile for NSBH and BNS-spinning on the documentation page all point to the BNS Makefile; please correct them.

\section meeting2014_10_22 October 10, 2014 meeting
\ref gstlalmeeting20141022page

iterutils.py: randindex()
This is in glue. Someone please move and address this

- Need a test code to show that the distribution produced is the intended one. Please produce some histograms of the distribution and attach to review documentation.

  - Beware of the change in LAL constants.


\section meeting2014_08_15 August 15, 2014 meeting
\ref gstlalmeeting20140815page

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

- Action: Jolien Creighton to focus on the delta function test.

- Action: Kipp/Chad -> Perform auto-correlation test.

- Action: Kipp/Chad -> Template bank test -> Do a bunch of injections and look at the distribution
of SNR/<SNR>.

- Action: Reviewers to interact with DQ team to make sure statevectors
are compatible.

<!---
- Write some unit tests for gstlal_drop
-->

\subsection sourcecodeadditions Action Items added to the following source codes

- gstlal_simulation

	- We need to look at this code again.

	- Please prepare a figure and a document to say what exactly
is being done to help complete the review.
	- Kipp will clean up this code and get it ready for review (Perhaps we could take one of the montly telecons to complete this review)

- gstlal_segmentsrc

	- Notes: Many changes were made to this code during the review. It
should be checked again and we should have a look at it again
at some point.

<!---
- gstlal_gate

	- Write a unit test.
-->

- gstlal_cachesrc

- gds_lvshmsrc


\section meeting2014_06_18 June 18, 2014 meeting
\ref gstlalmeeting20140618page

- Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.

\section meeting2014_04_09 April 9, 2014 meeting
\ref gstlalmeeting20140409page

- Chad: Please provide instructions on how to make the review pages appear online. At the moment the pages dont seem to appear even after several days and no one knows how to do this apart from Chad.
- The pages show up after the gstlalcbc account pulls and builds the doc.  This will be automatic once we move the doc to UWM nightly build
- Kipp: Prepare a simple flow-chart of the pipeline for the gstlal-inspiral analysis that produces GRACE-DB triggers. Identify the codes that are used in different boxes of the flow-chart and give us an idea of what those codes contain so we can together estimate the effort required to get the review done before aLIGO analysis.
- We have started to put some flow charts here for the low-latency analysis: \ref gstlalinspirallowlatencysearchpage
- Forent: Please run the injections using a sampling frequency > 16384 Hz and make sure it works.
- Kipp: The code allows injections with start times that are greater than end times. The code should not allow this to happen. Kipp to write a "if" statement check for start and end times and to fix this bug. (FIXED:  see 82db43aabc51ae5af1847771db60fb5438e8e546).
- Florent: The code does not say much about what is happening in the verbose mode: Run the code by using GST_DEBUG=lal_simulation:5, etc. to see if there is enough of debugging information.
- Florent:  Send Kipp instructions for reproducing the "only one injection when doing two on top of each other" demo
- Kipp: Explore why overlapping signal injections produced only one injection.
- Florent:  Send Kipp instructions for reproducing the "really slow and really really low sampling rate" demo
- Kipp: Explore why the codes runs slower with smaller sampling rates (e.g. 10 Hz as opposed to 1 kHz takes longer).


\section meeting2014_03_26 March 26, 2014 meeting
\ref gstlalmeeting20140326page

- Now that everyone is an expert on how to run gstlal_fake_frames how about the following: each reviewer take a usage case from the gstlal_fake_frames documentation
 - Sathya: Usage case 1
 - Jolien: Usage case 2
 - Duncan: Usage case 3
 - Florent: Usage case 4
	- each person should run the case and use the information in the "Debug" section of the documentation to write out the pipeline.  Also, try making plots of the output, etc.  Examine it critically and figure out what questions you have and what you would like to see answered in order to validate each piece.  We will go over each case next week.

\section meeting2014_03_05 March 5, 2014 meeting
\ref gstlalmeeting20140305page

- Kipp or Chad to create a separate gstlal_debug directory to host test codes and packges.
- Jolien to add his test package on lal_checktimestamps
- Fabien, Duncan and Sathya to install the test package
- Kipp should send instructions/pipeline to test gstlal_whiten
- Review should at some point go through XLALPSDRegressor codes (there quite a few but in this category that have been written solely for gstlal).

\section meeting2014_02_26 February 26, 2014 meeting
\ref gstlalmeeting20140226page

- Chad to fix lal_shift to properly set discont: Fixed in f9c5b20e1f2e13ad20d48da8ea83cbdf5a4d226f

\section meeting2014_02_19 February 19, 2014 meeting
\ref gstlalmeeting20140219page

<!---
- Jolien to write a unit test code for lal_checktimestamps
-->
- Chad has taken a stab at something that might help and checked it into gstlal/gstlal/tests.  This test program dynamically adds a one nanosecond time shift every time the user hits ctrl+C.  You need to do kill -9 to stop the program ;) Here is an example session

		$ ./lal_checktimestamps_test_01.py 
		src (00:00:05): 5 seconds
		^Cshifting by 1 ns
		lal_checktimestamps+lal_checktimestamps0: got timestamp 7.000666617 s expected 7.000666616 s (discont flag is not set)
		^Cshifting by 2 ns
		src (00:00:10): 10 seconds
		lal_checktimestamps+lal_checktimestamps0: got timestamp 10.000666618 s expected 10.000666617 s (discont flag is not set)
		lal_checktimestamps+lal_checktimestamps0: timestamp/offset mismatch:  got timestamp 10.000666618 s, buffer offset 20480 corresponds to timestamp 10.000666616 s (error = 2 ns)
		lal_checktimestamps+lal_checktimestamps0: timestamp/offset mismatch:  got timestamp 11.000666618 s, buffer offset 22528 corresponds to timestamp 11.000666616 s (error = 2 ns)	

	Note how the first ctrl+C only gives a warning since 1 ns is within the "fuzz".  But after the second ctrl+C there is an error. If this test is useful we can add it to the lal_checktimestamps documentation directly.  


\section meeting2014_01_20 January 20, 2014 Meeting
\ref gstlalmeeting20140120page

- update links in project page (springboard): done.
- make an example page to show reviewers how edit documentation, etc.: Email sent

Chads Actions
- Investigate the use of make_whitened_multirate_src() vs the whitener in gstlal_fake_frames
<!---
- lal_shift: make a unit test
-->
- Fix order of capsfilter / audioresample in gstlal_fake_frames graph Commit: b8d40a78d2484b32867ef06b7cc574871725f589
- Add dot graph output for gstlal_fake_frames Commit: b8d40a78d2484b32867ef06b7cc574871725f589
- make dot graph get dumped by env variable. Commit: 434a2d61eb5611817444309398d0859018dfed86


\section completed_action Completed action items

- Verify that all the segments are being tracked in online mode via the Handler (this is coupled to the inspiral.Data class, so it will come up again there) & Generate plots of the various segment output and come up with sanity checks.
	- *Done:* (not pushed yet, but PR open https://bugs.ligo.org/redmine/issues/2051)

- Background estimations should have more informative plots e.g., smoothed likelihood functions
	- *Done:* see 33b29f8b653c1bb10fdec477e05644ed6b46da0d 

- Test delta function input to LLOID algorithm (e.g with and without SVD)
	- *Done:* see \gstlalmeeting20150112page

- Consider how to let the user change SNR threshold consistently (if at all).  Note this is tied to SNR bins in far.py
- *Chad this will not be done right now*: The SNR threshold is tied to many
   histogramming objects.  Currently the value is set at 4 which is at the
saturation point for Gaussian noise, e.g., we expect to get an SNR 4 trigger
about once per second.  A user can  change this only after the histograming
saturation point for Gaussian noise, e.g., we expect to get an SNR 4 trigger
about once per second.  A user can  change this only after the histograming
code is generalized. It will be left for a future feature request.  Things to consider:
   - Change the histogram boundaries
   - Change the composite detection statistic threshold
   - Change the peak time on lal_itac
   - Study the dependence of coincidence triggers on SNR threshold
- Test robustness of fixed bank (start by figuring out the right question!)
 - Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.
 - *Marcel: Done* His thesis makes conclusions that it is okay: https://dcc.ligo.org/LIGO-L1400140

- Add synopses for all programs in documentation
 - *Chad: Done*
- Document offline pipeline including graphs of workflows
 - *Chad: Done*, see \ref gstlalinspiralofflinesearchpage
- Test pipeline with control peak times set to different values
 - *Chad: Done* see \ref gstlalinspiralcontrolpeaktimestudypage

- put iterutils.randindex() test plot into minutes for 2014-11-05 telecon

- added check to svd_bank that the command line option matches snr_min in
  far.py

- added test of behaviour of sampler generators to glues test suite

- expanded docstring for NDBins.volumes() so that this method is covered by
  pylals test suite

- removed ImportError path for speed of light from snglcoinc.py

- see if P(instruments | signal) can be included in P(snr, ... |
  instruments):  yes it can be, like the SNR PDFs it depends only on the
  ratios of the horizon distances;  have not made this improvement yet but
  have put a FIXME in the code to remind me that this is how to achieve the
  tracking of the time dependence of the instrument combination factor

- see if the noise coinc rates can be recomputed on the fly by factoring
  out the rates:  yes, they can be;  have not made this improvement because
  it would require tinkering with the code a bit and we are about to do a
  release, but I have added a FIXME to remind me that this performance
  improvement is possible.

nightly build:
	- turned off SL6 until it works
	- got nightly build running on debian:
		- includes all gstlal-packages, all documentation, all unit tests

- Analysis Makefiles should be documented (e.g., parameters); Do we want them to be made more generic?
 - *Chad: Done*.  \ref gstlalinspiralofflinesearchpage

- Write joint likelihood ranking and FAP calculation (paper in progress)
 - *Kipp: Done* LIGO P1400175 http://arxiv.org/abs/1504.04632

- show histogram of horizon distance history
 - *Cody: Done* 54368d058460d37473e78bf26776a4929db01433
