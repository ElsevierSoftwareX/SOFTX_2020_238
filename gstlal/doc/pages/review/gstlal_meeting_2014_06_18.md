\page gstlalmeeting20140618page Review Meeting June 18, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

 - See completed action items here: \ref gstlalinspirallowlatencysearchpage 
 - Suggest we continue with python codes: multirate_datasource.py cbc_template_fir.py inspiral_pipe.py
 - Suggest we continue with the whitener element

\section minutes minutes

General Action Items: 

 - Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.


\subsection multirate Code review: mkwhitened_multirate_src

 - Resamplers (really downsamplers) are missing in the graph of

 -Is the h(t) gate really necessary? It shouldnt really be used unless
there is something really wrong with the data. Wishlist: Tee off from 
the control panel and record on/off (This is already done).

 -Task for the review team: Check what data was analysed and how much
data was "lost" due to application of internal data quality.

 -  There seems to be a bug in resampler (even) in the latest version of gstreamer; 
(produces one few sample). We need to better understand the consequence of this bug.

\section inspiral_pipe Code review: inspiral_pipe.py

 - In inspiral_pipe.py Fix the InsiralJob.___init___: fix the arguments

 - On line 201, fix the comment or explain what the comment is meant to be

\subsection cardiff Status of Action Items from Cardiff meeting 

 -# Test robustness of template bank: Ongoing tests by Marcel Kehl (grad student at CITA). 
 -# Analysis Makefiles should be documented: Started
 -# Test delta function input to LLOID algorithm: Ongoing at UWM.

4. Synpses of all programs in documentation: Done.

5. Background estimations should have more informative plots e.g., smoothed 
likelihood functions: Ongoing at Penn State.

6. Study the dependence of coincidence triggers on SNR threshold
Document offline pipeline including graphs of workflows
Chad: Done, see Offline search documentation

7. Test pipeline with control peak times set to different values
Chad: Done see Study of the composite detection statistic, (i.e., 
control-peak-time)

8.  Write documentation for autochisq (paper in progress)

9.  Write joint likelihood ranking and FAP calculation (paper in progress)

10.  Explore autocorrelation chisquared mismatch scaling with number of 
samples e.g., $ \nu + \epsilon(\nu) \delta^{2} $

11.  Make sure capsfilters and other similar elements are well documented 
within graphs (e.g. put in rates, etc)

12.  Add description of arrows in graphs

13.  Verify that all the segments are being tracked in online mode via 
the Handler (this is coupled to the inspiral.Data class, so it will come 
up again there)

14.  Feature request for detchar - It would be helpful to have online 
instrument state that could be queried to know if an instrument will 
be down for an extended time


---------------------------
Review of: multirate_datasource.mkwhitened_multirate_src
o The vetos applied will not be the same for the online and offline analysis.

o Florent says we should never use an h(t) veto, unless there are some large
SNR excursions ~ 10,000, e.g.

o Is it possible to move this out of gstlalinspiral and put it in 
gstlalmkstream?

o There seems to be a bug in resampler (even) in the latest version of gstreamer; 
(produces one few sample). We need to better understand the consequence of this bug.
Review Status: OK with minor actions. 

mkwhitened_multirate_src is reviewed with comments:
---------------------------

Review of: inspiral_pipe.py 
Fix the InsiralJob.___init___: fix the arguments

Review Status: OK with minor actions. 
