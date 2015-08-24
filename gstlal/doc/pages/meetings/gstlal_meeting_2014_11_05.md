\page gstlalmeeting20141105page Review Meeting November 05, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- Code review links for telecon:
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/glue/glue/iterutils.py
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/pylal/pylal/rate.py
 - https://ligo-vcs.phys.uwm.edu/cgit/lalsuite/tree/pylal/pylal/snglcoinc.py
 - https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/python/far.py
- Discussion of offline DAG
 - \ref gstlalinspiralofflinesearchpage


\section minutes minutes

In attendance: Hanna, Cannon, Meacher, Creighton J, Robinet, Sathyaprakash, Messick, Dent, Blackburn

<!---
\section action Action Items:General


1. Chad: To amend the documentation to warn about fixed SNR threshold of 4.  At some point in the future consider making SNR threshold an adjustable parameter

2. Kipp: Need to fix distance quantization for the construction of SNR PDFs

3. Chad: You have a hard-coded 4096 Hz, find out where that is; need to have a variable sampling rate (perhaps this is not hardcoded any more???).

4. Chad: Time-dependence of horizon distances can be captured in the computation of the 'numerator'

\section Action Items:Makefile:

5. Chad: Indicate that overlap should be 10% of the num_split_templates

6. Chad: People shouldn't have to remove Multicore=True

7. Chad:  The length of autocorrelation chi-squared in sample points is set to AC_LENGTH = 351. It might be worth exploring how the sensitivity volume changes by varying AC_LENGTH

8. Chad: Please specify the boundaries of the various parameters in Makefile, especially if it is desirable to explore how the sensitivity changes for different values of these parameters

9. Chad: Makefile for NSBH and BNS-spinning on the documentation page all point to the BNS Makefile; please correct them.

\section minutes minutes

- Looked at the "random sampling" unit test. The output graph looks OK. Please add the output to the review page.

snglconic.py:
- Looked at init function and dictionary

far.py:
- It would be good to histogram the distances as a function of GPS times

- Sathya: How many places is the SNR threshold is hard-coded?

- Kipp/Chad: In three places, DAG script, this code and binning

- Chad: Motivation for SNR threshold of 4 was to get 1 trigger per second and so it won't be any worse off even if templates are short and look like glitches

- Kipp: For the early release use a fixed SNR PDF using median PSD which could be pre-computed and stored on disk. (Store database of SNR pdfs for a variety of horizon)

- Jolien: Could use simple scaling.
- Sathya: Plots of SNR in different instruments don't really look simple enough to do simple scaling

- Florent: The binning parameters are hard-coded too; Could it be a problem
- Kipp: (What was your precise response Kipp?)

- Jolien: Line: 640: Why not use finish() method inside iadd().
- Kipp: It is too time consuming

- A bunch of questions on how the likelihood is calculated. We need to better understand this; to be presented at the Rates and Significance call next week.

- Jolien: What changes when you go to high masses is not the prefactor but the number of bins.

- Jolien: Suggests that time-dependence of horizon distances can be captured in the computation of the "numerator"

- Jolien: Example in 1165 could print the output.

- Kipp: Chisquare binning hasn't been tuned to be a good representation of the PDFs; could be improved in future
-->

Makefiles for various runs:

- Kent: what changes are needed to get a new approximant to work:
- Chad: Make changes to line 65/66 templates.py

- Note that some of the variables in the Makefile are defined in Makefile.offline_analysis_rules

- \--sort can choose almost any parameters in the template bank params (masses, chirp mass, etc.) Changes would be needed in gstlal__bank__splitter_source if you need to sort on other parameters.

- \--bank-cache You would need to know the string that specified the way the bank was made

- \--peak-time used to be 8 s but for MDC it is turned off (i.e. uses 0) (computational cost could be reduced by tens of percents by using a peak time of 4s or 8s without significantly changing the efficiency).
