\page gstlalinspirals6replaypage S6 Replay Documentation

[TOC]

\section Introduction Introduction

With O1 quickly approaching in mid to late 2015, the CBC group is doing a live
data simulation run using S6 data to test piplines for the purpose of review.
This documentation page is specific to the gstlal portion of the simulation
run.  

\subsection Goals Goals

 - Establish the accuracy of False Alarm Rate/False Alarm Probability (FAR/FAP) calculations in online analysis for low mass systems
 - Establish the online analysis has the appropriate sensitivity
 - Ensure that the information provided to gracedb for events is sufficient for EM alerts and for collaboration consumption.

\subsection Proposal Proposed Approach

The gstlal analysis team proposes to use one month of S6 data replayed in an
online environment.  Details can be found
<a href="https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/140812103550GeneralData%20broadcasting">here</a>

\section Data One Month Data Details

Some quick facts:

 - Data GPS Start: 968543943 -> 1120766224
 - Data GPS End:   971622087 -> 1125226624
 - IFOs: H1, L1
 - Analysis GPS Start: ~1122174187
 - Broadcast status: http://soapbox.cgca.uwm.edu:33655/index_auto_reload.html


\subsection Resources Resources

 - Online: 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002
 - Offline: CIT Cluster (does not need to be as specific as online)


\subsection AnalysisCodes Analysis Codes

 - gstlal: 7466e2bfe87adef273574e9068eced0be683dc9a
 - lalsuite: 3dc971a085afdbf06b44fb463ed08036270bf377

\subsection Injections Injection Parameters

 - Component mass normally distributed with mean mass of 1.4 \f$M_\odot\f$ and standard deviation of 0.01 \f$M_\odot\f$
 - Sources uniformly distributed in interval [5,45] Mpc
 - Spinless
 - Injected every 20 &plusmn; 3 seconds
 - https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Software%20Online%20injections#injections_parameters

\subsection Online Online Analysis

\subsubsection OnlineBanks Template Banks

 - /home/gstlalcbc/review/s6replay/online/bank
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay_bank>Makefile to make the template banks</a>

\subsubsection OnlineTriggers Online Triggers

 - /home/gstlalcbc/review/s6replay/online/trigs
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay_online>Makefile to make the analysis dag</a>

\subsection Offline Offline Analysis

 - At CIT
 - /home/gstlalcbc/s6replay_bigdog/offline_one_month_no_stochastic_injections
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay> Makefile </a>

\subsection Results Results

 - <a href=https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary> Online Status Page </a>
 - <a href="https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/ALL_LLOID_COMBINED_openbox.html">Online Summary Page</a>
 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201122174187..1124174187">GraceDb query</a>
 - <a href="https://simdb.cgca.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201122174187..1124174187">SimDb query</a>
 - <a href="https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/offline_s6_replay_1monrun/">Offline Analysis Page</a>


\subsection status Status As Of July 31 

\subsubsection nagios Nagios Monitor

https://dashboard.ligo.org/

\subsubsection condor Condor Status

	-- Submitter: pcdev3.nemo.phys.uwm.edu : <192.168.5.3:41523> : pcdev3.nemo.phys.uwm.edu
	 ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD               
	1708701.0   gstlalcbc       7/28 22:02   2+10:32:54 R  0   0.3  condor_dagman -f -
	1708708.0   gstlalcbc       7/28 22:02   2+10:32:36 R  0   0.0  gstlal_inspiral_ma
	1708709.0   gstlalcbc       7/28 22:02   2+10:32:36 R  0   0.0  gstlal_ll_inspiral
	1708710.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   9765.6 gstlal_inspiral --
	1708711.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   7324.2 gstlal_inspiral --
	1708712.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   9765.6 gstlal_inspiral --
	1708713.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   7324.2 gstlal_inspiral --
	1708714.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   9765.6 gstlal_inspiral --
	1708715.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   7324.2 gstlal_inspiral --
	1708716.0   gstlalcbc       7/28 22:02   2+10:32:21 R  0   7324.2 gstlal_inspiral --
	1708717.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708718.0   gstlalcbc       7/28 22:03   2+10:32:21 R  0   7324.2 gstlal_inspiral --
	1708719.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708720.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   9765.6 gstlal_inspiral --
	1708721.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708722.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708723.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708724.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   9765.6 gstlal_inspiral --
	1708725.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   7324.2 gstlal_inspiral --
	1708726.0   gstlalcbc       7/28 22:03   2+10:32:20 R  0   9765.6 gstlal_inspiral --
	1708727.0   gstlalcbc       7/28 22:03   2+10:31:55 R  0   7324.2 gstlal_inspiral --
	1708728.0   gstlalcbc       7/28 22:03   2+10:31:54 R  0   7324.2 gstlal_inspiral --
	1708729.0   gstlalcbc       7/28 22:03   2+10:31:55 R  0   7324.2 gstlal_inspiral --
	1708730.0   gstlalcbc       7/28 22:03   2+10:31:54 R  0   9765.6 gstlal_inspiral --
	1708731.0   gstlalcbc       7/28 22:03   2+10:31:54 R  0   9765.6 gstlal_inspiral --
	1708732.0   gstlalcbc       7/28 22:03   2+10:31:54 R  0   9765.6 gstlal_inspiral --
	1708733.0   gstlalcbc       7/28 22:03   2+10:31:54 R  0   9765.6 gstlal_inspiral --
	1708735.0   gstlalcbc       7/28 22:03   2+10:32:05 R  0   0.0  lvalert_listen --u
	1714098.0   gstlalcbc       7/30 08:27   1+00:08:29 R  0   0.0  gstlal_ll_inspiral


\section Review  Review


\subsection day1  July 29, 2015

In attendance: Chad, Edward Maros, Gareth Tomas, Jolien Creighton, Kent Blackburn, Laleh Sadeghian, Les Wade, Maddie Wade, Sathyaprakash, Sarah Caudill, Tjonnie Li


\subsubsection day1actions Actions

 - Produce b/g plots with and without the candidates

 - If possible label on the contours on the background plots
 
 - Money plot here looks strange (dominated by priors, insufficient data, ...);
  it would be good if release versions can fix this:
<https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/Images/H1L1V1-GSTLAL_INSPIRAL_PLOTSUMMARY_ALL_LLOID_COMBINED_05_count_vs_ifar_openbox-1121652724-550359.png>.
One issue is that the clustering used in gstlal_inspiral internally in online
running is set by the parameter "thinca-interval", which we usually set to 1
second for latency reasons.  That means that triggers are clustered over a 1
second interval.  The offline clustering SQL file uses a &plusmn; 4s window.
So I modified that script to use a &plusmn; 0.5s window, which I hope will
approximate the gstlal_inspiral internal clustering well enough.  See:
88cebe1656e267fd9d6579db83e4cdfff00c804e.  This was not however sufficient to
fix the IFAR plot. I still think that this is simply due to the fact that there
has not been sufficient data to overwhelm the very conservative prior.  Lets
see if the situation improves as we go forward.


\subsubsection day1completed Completed Actions

 - fix the missed found plot in the iHope pages: <https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/Images/H1L1V1-GSTLAL_INSPIRAL_PLOTSUMMARY_ALL_LLOID_COMBINED_01_deff_vs_t_H1L1_closedbox-1121652724-550359.png>. Done, see: 1ea0df6af5ec219db95c0784ca186e40acfb91d0
 
 - Remove candidate event(s) from closed box results. Done see: a886f41349ee2369615cb49455dfcfb7695bcc7a

 - Make the information about injection rate available on the summary pages along side the parameters used for injections: Done see above
 
 - Instead of overwriting online iHope pages it would be good to save the old one (with a time stamp) and start afresh but with all the data so that we can have the history available to look at. Done, see: 43cc68e42c882a4710196ebe51633432b0ad2bca

 - Use a smoothing kernel with a larger (try 2, 4) scale to smoothen the likelihood and probability plots. This has been investigated and is documented here: https://bugs.ligo.org/redmine/issues/2339



\subsection day2  July 30, 2015

In attendance: Chad, Jolien and Sathya

\subsubsection day2actions Actions

 - Look at missed close-by injections (there are not many but it is puzzling why ~8 Mpc injections are missed; could be due to injections being on lock-loss boundary but should be investigated).

 - Use integer GPS second for the on-line clustering script.

 - The SNR heat plot and the SNR-Chisquare trigger plots do not appear in SIM-DB page; it would be good to have them there too to compare with the Big Dog event and to see if the anatomy of the triggers is similar when there is a loud event. 


\subsubsection day2completed Completed Actions
