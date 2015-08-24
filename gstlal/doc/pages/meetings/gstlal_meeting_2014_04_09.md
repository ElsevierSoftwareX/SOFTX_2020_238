\page gstlalmeeting20140409page Meeting on April 09, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- Code review goals: Now that we have some experience with gstlal codes we took stock of what is that we need to review. Everyone on the call agreed that the most important critical aspect for the review is the gstlal pipeline that produces tiggers that are sent to Grace-DB which are in turn sent for EM follow-up but, more importantly, our first source of detection candidates. The team should focus in reviewing codes that are used to produce Grace-DB triggers (see below for actions).

- Discuss gstlal_fake_frames runs by Florent and Duncan (only got through Florent; run by Duncan next week).


<!---
Actions
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
-->

\section questions Questions

1. Timing? I checked the time of the injections taking into account the time propagation to the detectors. --> OK

2. Shifting? I added a time_slide table and checked that the injections were correctly time-shifted. --> OK

3. Edge effect? An injection has been made on the edge of the frame time range. It has been correctly performed (and truncated). --> OK

4. Sampling? I tested different sampling rates. The result is robust against that.

5. I noticed that the lower the sampling rate the slower the program runs. I would expect the opposite behavior...

6. why the sampling rate cannot exceed 16384 Hz? I think this requirement is too strong. There are many examples where we need to sample with higher rates: detchar aux. channels, Virgo h(t)...

7. Question: is there some windowing happening when doing the injection? I guess there must be one: could you explain how exactly?

8. bug? I performed twice the same injection but shifted by 0.2s. I would expect to see the 2 resulting injections overlapping in time. As you can see on the zoom plot, only one peak is visible. Where is the second injection?

9. comment: I used the --verbose option but I found the verbosity to be very low. I think for debugging purposes, it would be nice to have more information.

10. I ran gstlal_fake_frames with a gps end time *before* the gps start time and... it worked! What does it mean?
