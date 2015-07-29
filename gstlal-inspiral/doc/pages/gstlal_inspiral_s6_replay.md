\page gstlalinspirals6replaypage S6 Replay Documentation

[TOC]

\section Introduction Introduction

With O1 quickly approaching in mid to late 2015, the CBC group is doing a live
data simulation run using S6 data to test various piplines for the purpose of
code review.  This documentation page is specific to the gstlal portion of the
simulation run.  

\subsection Goals Goals

 - Establish the accuracy of False Alarm Rate/False Alarm Probability (FAR/FAP) calculations in online analysis for low mass systems
 - Establish the online analysis has the appropriate sensitivity

\subsection Proposal Proposed Approach

The gstlal analysis team proposes to use two weeks of S6 data replayed in an
online environment.  Details can be found
<a href="https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/140812103550GeneralData%20broadcasting">here</a>

\section Data 1 Month Data

Some quick facts:

 - Data GPS Start: 968543943 -> 1120766224
 - Data GPS End:   971622087 -> 1125226624
 - IFOs: H1, L1
 - Analysis GPS Start: ~1122174187
 - Broadcast status: http://soapbox.cgca.uwm.edu:33655/index_auto_reload.html


\subsection Resources Resources

 - Online: 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002
 - Offline: CIT Cluster (does not need to be as specific as online)


\subsection AnalysisCodes Analysis codes

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

 - <a href=https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary> online status page </a>
 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201122174187..1124174187">GraceDb query</a>
 - <a href="https://simdb.cgca.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201122174187..1124174187">SimDb query</a>
 - <a href="https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/offline_s6_replay_1monrun/">Offline analysis results</a>


\subsection status status as of 1122210960 

	-- Submitter: pcdev3.nemo.phys.uwm.edu : <192.168.5.3:41523> : pcdev3.nemo.phys.uwm.edu
	 ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD               
	1708701.0   gstlalcbc       7/28 22:02   0+10:13:07 R  0   0.3  condor_dagman -f -
	1708707.0   gstlalcbc       7/28 22:02   0+10:12:49 R  0   0.0  gstlal_ll_inspiral
	1708708.0   gstlalcbc       7/28 22:02   0+10:12:49 R  0   0.0  gstlal_inspiral_ma
	1708709.0   gstlalcbc       7/28 22:02   0+10:12:49 R  0   0.0  gstlal_ll_inspiral
	1708710.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   4882.8 gstlal_inspiral --
	1708711.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   4150.4 gstlal_inspiral --
	1708712.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   7324.2 gstlal_inspiral --
	1708713.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   4638.7 gstlal_inspiral --
	1708714.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   7324.2 gstlal_inspiral --
	1708715.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   4394.5 gstlal_inspiral --
	1708716.0   gstlalcbc       7/28 22:02   0+10:12:34 R  0   7324.2 gstlal_inspiral --
	1708717.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   4394.5 gstlal_inspiral --
	1708718.0   gstlalcbc       7/28 22:03   0+10:12:34 R  0   4638.7 gstlal_inspiral --
	1708719.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   3906.2 gstlal_inspiral --
	1708720.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   4882.8 gstlal_inspiral --
	1708721.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   4638.7 gstlal_inspiral --
	1708722.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   7324.2 gstlal_inspiral --
	1708723.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   4882.8 gstlal_inspiral --
	1708724.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   7324.2 gstlal_inspiral --
	1708725.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   7324.2 gstlal_inspiral --
	1708726.0   gstlalcbc       7/28 22:03   0+10:12:33 R  0   7324.2 gstlal_inspiral --
	1708727.0   gstlalcbc       7/28 22:03   0+10:12:08 R  0   4882.8 gstlal_inspiral --
	1708728.0   gstlalcbc       7/28 22:03   0+10:12:07 R  0   7324.2 gstlal_inspiral --
	1708729.0   gstlalcbc       7/28 22:03   0+10:12:08 R  0   7324.2 gstlal_inspiral --
	1708730.0   gstlalcbc       7/28 22:03   0+10:12:07 R  0   7324.2 gstlal_inspiral --
	1708731.0   gstlalcbc       7/28 22:03   0+10:12:07 R  0   7324.2 gstlal_inspiral --
	1708732.0   gstlalcbc       7/28 22:03   0+10:12:07 R  0   7324.2 gstlal_inspiral --
	1708733.0   gstlalcbc       7/28 22:03   0+10:12:07 R  0   7324.2 gstlal_inspiral --
	1708735.0   gstlalcbc       7/28 22:03   0+10:12:18 R  0   0.0  lvalert_listen --u


\section TwoWeeks Two Week testing run


\subsection TwoWeekData Two Week Data

Some quick facts:

 - GPS Start: 967161687 -> 1119131821
 - GPS End:   968371287 -> 1120341421
 - IFOs: H1, L1


\subsection twoweekstatus status as of Jul 1

		1573929.0   gstlalcbc       6/23 17:45   7+15:39:10 R  0   0.0  gstlal_inspiral_ma
		1573930.0   gstlalcbc       6/23 17:45   7+15:39:05 R  0   0.0  gstlal_ll_inspiral
		1573931.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573932.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573933.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   9765.6 gstlal_inspiral --
		1573934.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573935.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   9765.6 gstlal_inspiral --
		1573936.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573937.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   9765.6 gstlal_inspiral --
		1573938.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573939.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   9765.6 gstlal_inspiral --
		1573940.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573941.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   9765.6 gstlal_inspiral --
		1573942.0   gstlalcbc       6/23 17:45   7+15:38:42 R  0   7324.2 gstlal_inspiral --
		1573943.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   9765.6 gstlal_inspiral --
		1573944.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573945.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   9765.6 gstlal_inspiral --
		1573946.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573947.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   9765.6 gstlal_inspiral --
		1573948.0   gstlalcbc       6/23 17:45   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573949.0   gstlalcbc       6/23 17:46   7+15:38:41 R  0   9765.6 gstlal_inspiral --
		1573950.0   gstlalcbc       6/23 17:46   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573951.0   gstlalcbc       6/23 17:46   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573952.0   gstlalcbc       6/23 17:46   7+15:38:41 R  0   7324.2 gstlal_inspiral --
		1573953.0   gstlalcbc       6/23 17:46   7+15:38:41 R  0   9765.6 gstlal_inspiral --
		1573954.0   gstlalcbc       6/23 17:46   7+15:38:14 R  0   9765.6 gstlal_inspiral --
		1573955.0   gstlalcbc       6/23 17:46   7+15:38:38 R  0   0.0  lvalert_listen --u


\section Results Results

 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201119131821..1120341421">GraceDb query</a>
 - <a href="https://simdb.phys.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201119131821..1120341421">SimDb query</a>


\section Review


\subsection day1  July 29, 2015

In attendance: Chad, Edward Maros, Gareth Tomas, Jolien Creighton, Kent Blackburn, Laleh Sadeghian, Les Wade, Maddie Wade, Sathyaprakash, Sarah Caudill, Tjonnie Li


\subsubsection day1actions Actions

 - Produce b/g plots with and without the candidates

 - Use a smoothing kernel with a larger (try 2, 4) scale to smoothen the likelihood and probability plots.

 - If possible label on the contours on these plots

 - Money plot here looks strange (dominated by priors, insufficient data, ...); it would be good if release versions can fix this: <https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/Images/H1L1V1-GSTLAL_INSPIRAL_PLOTSUMMARY_ALL_LLOID_COMBINED_05_count_vs_ifar_openbox-1121652724-550359.png>.  Note, I think this is the expected behavior, lets wait to see how it evolves during the replay

\subsubsection day1completed Completed Actions

 - fix the missed found plot in the iHope pages: <https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/Images/H1L1V1-GSTLAL_INSPIRAL_PLOTSUMMARY_ALL_LLOID_COMBINED_01_deff_vs_t_H1L1_closedbox-1121652724-550359.png>. Done, see: 1ea0df6af5ec219db95c0784ca186e40acfb91d0
 
 - Remove candidate event(s) from closed box results. Done see: a886f41349ee2369615cb49455dfcfb7695bcc7a

 - Make the information about injection rate available on the summary pages along side the parameters used for injections: Done see above
 
 - Instead of overwriting online iHope pages it would be good to save the old one (with a time stamp) and start afresh but with all the data so that we can have the history available to look at. Done, see: 43cc68e42c882a4710196ebe51633432b0ad2bca
