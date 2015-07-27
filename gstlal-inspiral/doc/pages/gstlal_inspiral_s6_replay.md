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

\section Proposal Proposed Approach

The gstlal analysis team proposes to use two weeks of S6 data replayed in an
online environment.  Details can be found
<a href="https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/140812103550GeneralData%20broadcasting">here</a>

\subsection Data Data

Some quick facts:

 - GPS Start: 967161687 -> 1119131821
 - GPS End:   968371287 -> 1120341421
 - IFOs: H1, L1

\subsection Resources Resources

 - Online: 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002
 - Offline: NEMO Cluster (does not need to be as specific as online)

\section Analysis Analysis

 - online UWM: /home/gstlalcbc/review/s6replay/online/trigs
 - offline UWM: /home/gstlalcbc/review/s6replay/offline/

\subsection AnalysisCodes Analysis codes

 - gstlal 1a44f7af0cf69293f4b0883e4e4142ce263e86f4
 - all other dependencies from ER7 releases

\subsection Injections Injection Parameters

 - Component mass normally distributed with mean mass of 1.4 \f$M_\odot\f$ and standard deviation of 0.01 \f$M_\odot\f$
 - Sources uniformly distributed in interval [5,45] Mpc
 - Spinless

\subsection Online Online Analysis

\subsubsection OnlineBanks Template Banks

 - /home/gstlalcbc/review/s6replay/online/bank
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay_bank>Makefile to make the template banks</a>

\subsubsection OnlineTriggers Online Triggers

 - /home/gstlalcbc/review/s6replay/online/trigs
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay_online>Makefile to make the analysis dag</a>

\subsection Offline Offline Analysis

 - /home/gstlalcbc/review/s6replay/offline
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay>Makefile which contains rules for every offline analysis</a>

\section status status as of Jul 1

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
 - <a href=https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary?id=0000,0011&dir=/home/gstlalcbc/review/s6replay/online/trigs&ifos=H1,L1> online summary page </a>
 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201119131821..1120341421">GraceDb query</a>
 - <a href="https://simdb.phys.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201119131821..1120341421">SimDb query</a>
 - <a href="https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/range.png">Low latency sensitivity plots</a>
   - Covers the last 48 hours
   - Updated every 5-10 minutes
 - <a href="https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/offline_s6_replay/">Offline analysis results</a>
