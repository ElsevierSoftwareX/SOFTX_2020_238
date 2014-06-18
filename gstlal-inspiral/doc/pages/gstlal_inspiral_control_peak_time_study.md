\page gstlalinspiralcontrolpeaktimestudy Study of the composite detection statistic, (i.e., control-peak-time)

\section intro Introduction

The purpose of this page is to investigate the effect of using a composite detection statistic described in <a href=http://arxiv.org/abs/1101.0584>arxiv:1101.0584</a> to first identify interesting places to reconstruct SNR.  The idea is that it can save floating point operations.

\section method Method

*This is preliminary and shouldn't be considered a final answer about whether or not this is a good idea.*

Two gstlal_inspiral analyses were run.  The first did not use a control peak time.  The second did not, e.g.,

		channa@pcdev1:~/MDC_new/test$ diff -u Makefile ../test_2s/Makefile
		--- Makefile	2014-06-17 20:35:16.969869229 -0500
		+++ ../test_2s/Makefile	2014-06-17 20:32:55.203392803 -0500
		@@ -17,10 +17,10 @@
		 IFOS = H1 L1 V1
		 START = 966384015
		 STOP = 967384015
		-TAG = T1200307_LV_gaussian_0s_w_zerolag_injections_40Hz_5ms_test
		+TAG = T1200307_LV_gaussian_2s_w_zerolag_injections_40Hz_5ms_test
		 WEBDIR = ~/public_html/MDC_new/${START}-${STOP}-${TAG}
		 NUMBANKS = 4
		-PEAK = 0
		+PEAK = 2
		 AC_LENGTH = 351
		 # additional options, e.g.,
		 #ADDITIONAL_DAG_OPTIONS = "--blind-injections BNS-MDC1-WIDE.xml"

\section results Results

The results are here:

 - <a href='https://ldas-jobs.phys.uwm.edu/~channa/MDC_new/966384015-967384015-T1200307_LV_gaussian_0s_w_zerolag_injections_40Hz_5ms_test/gstlal-966384015-967384015_open_box.html'> No control peak time</a>
 - <a href='https://ldas-jobs.phys.uwm.edu/~channa/MDC_new/966384015-967384015-T1200307_LV_gaussian_2s_w_zerolag_injections_40Hz_5ms_test/gstlal-966384015-967384015_open_box.html?gstlal-966384015-967384015_open_box_missed_found.html'>With 2s peak time</a>

The range plots are shown here:

@image html 0sVT.png "Range in Mpc vs FAR for no control peak time"

@image html 2sVT.png "Range in Mpc vs FAR for control peak time of 2 seconds"
