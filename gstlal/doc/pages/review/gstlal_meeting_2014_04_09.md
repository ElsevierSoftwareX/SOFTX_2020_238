\page gstlalmeeting20140409page Meeting on April 09, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- Discuss gstlal_fake_frames runs by Fabien and Duncan

\section homework Homework

Actions
 - Sathya: Usage case 1 (accomplished Usages 1-3 but not tested the output yet)
 - Jolien: Usage case 2
 - Duncan: Usage case 3
 - Florent: Usage case 4 (successfully ran several examples; see additional questions below)

\section questions Questions

1/ Timing? I checked the time of the injections taking into account the time propagation to the detectors. --> OK

2/ Shifting? I added a time_slide table and checked that the injections were correctly time-shifted. --> OK

3/ Edge effect? An injection has been made on the edge of the frame time range. It has been correctly performed (and truncated). --> OK

4/ Sampling? I tested different sampling rates. The result is robust against that.

5/ I noticed that the lower the sampling rate the slower the program runs. I would expect the opposite behavior...

6/ why the sampling rate cannot exceed 16384 Hz? I think this requirement is too strong. There are many examples where we need to sample with higher rates: detchar aux. channels, Virgo h(t)...

7/ Question: is there some windowing happening when doing the injection? I guess there must be one: could you explain how exactly?

8/ bug? I performed twice the same injection but shifted by 0.2s. I would expect to see the 2 resulting injections overlapping in time. As you can see on the zoom plot, only one peak is visible. Where is the second injection?

9/ comment: I used the --verbose option but I found the verbosity to be very low. I think for debugging purposes, it would be nice to have more information.

10/ I ran gstlal_fake_frames with a gps end time *before* the gps start time and... it worked! What does it mean?
