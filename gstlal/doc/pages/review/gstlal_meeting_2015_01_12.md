\page gstlalmeeting20150112page F2F Review Meeting January 12-18, 2015, Caltech

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

  - Look at MDC results for BNS
    - https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/BNS/MDC/SpinMDC/gstlal_pipe_compare
  - Look at S6-replay
  - Run pipeline tutorial
  - GRACE-DB: https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online_analysis.html

\section action Action Items

  - Compute the actual expected SNR (instead of from average PSD) and plot the SNR histograms again
  - We should compare the on-line and off-line results and understand the similarities and differences
  - Figure out why multiple low-SNR events get the same likelihood (these constitute about a quarter of all events)
  - Review of Summary pages at:
    <https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/MDC/BNS/Summer2014/recolored/nonspin/966384015-971384015-pipe-compare-CAT2/ALL_LLOID_COMBINED_openbox.html?ALL_LLOID_COMBINED_openbox_summary.html>
    - Make zoomed versions for accuracy plots
    - Make Histogram of accuracies
    - Make accuracy plots as a function of FAR 
    - Plot accuracy as a function of SNR
    - Injections are found with "pi" time-shift; probably coming from one time-slide (check and fix)
    - What is the reason for close-by missed injections in missed-found plot (as a function of Mchirp)?
    - Perpahs the prefactors_range (0.0, 0.10) might be too narrow 
    - Figure out the reason for the dip in the foreground plot (in SNR-Chisq section)
    - Make efficiency plots (efficiency as a function of distance, effective distance and chirp distance)
    - Compute the Range as a function of network for an IDEAL pipeline (T1200458, Figure 2 would be a good example)
    - Update the online instructions including Makefile

  - Review of GRACE-DB:
    - Check live times.
    - It would be useful to have a Table of CPU usage rather than for individual nodes.
    - Currently, on-line analysis requests a headroom of 20% CPU. Can this be defended? Explore running some "nice" jobs in the background and see if this affects performace 
    - The particular event we saw had a 17 minute latency for producing sky-map: https://gracedb.ligo.org/events/view/T124866


\section minutes minutes

  - For online search one can look at dashboard.ligo.org to see if a given detector is producing data
  - Live time etc. can be found at: https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online_analysis_node03.html
  - The review committee should come up with a list of items that should go on the summary pages on GRACE-DB. (Consult with Detection Committee and the CBC group).
  - 
