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

 - Data GPS Start: 968543943 -> 1125803595
 - Data GPS End:   971622087 -> 1128881739
 - IFOs: H1, L1
 - Analysis GPS Start: ~1125838327
 - Broadcast status: http://soapbox.cgca.uwm.edu:33655/index_auto_reload.html


\subsection Resources Resources


 - Online: 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002
 - Offline: CIT Cluster (does not need to be as specific as online)


\subsection AnalysisCodes Analysis Codes


 - gstlal: 
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

\subsubsection OnlineTriggers Online Triggers

 - /home/gstlalcbc/review/s6replay/online/O1_release_trigs


\subsection Offline Offline Analysis


 - At CIT
 - /home/gstlalcbc/s6replay_bigdog/offline_one_month_no_stochastic_injections
 - <a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/tree/gstlal-inspiral/share/Makefile.s6_replay> Makefile </a>


\subsection Results Results


 - <a href=https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary> Online Status Page </a>
 - <a href="https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/ALL_LLOID_COMBINED_openbox.html">Online Summary Page</a>
 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201125803595.0..1128881739.0">GraceDb query</a>
 - <a href="https://simdb.cgca.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201125803595.0..1128881739.0">SimDb query</a>
 - <a href="https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/offline_s6_replay_1monrun/">Offline Analysis Page</a>


\subsection status Status As Of August 4


\subsubsection nagios Nagios Monitor

https://dashboard.ligo.org/

\subsubsection condor Condor Status

	-- Submitter: pcdev3.nemo.phys.uwm.edu : <192.168.5.3:51007> : pcdev3.nemo.phys.uwm.edu
	 ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD               
	1916113.0   gstlalcbc       9/9  07:47   0+02:50:23 R  0   0.3  condor_dagman -f -
	1916116.0   gstlalcbc       9/9  07:47   0+02:50:08 R  0   0.0  gstlal_ll_inspiral
	1916117.0   gstlalcbc       9/9  07:47   0+02:50:08 R  0   0.0  gstlal_ll_inspiral
	1916118.0   gstlalcbc       9/9  07:47   0+02:50:08 R  0   0.0  gstlal_inspiral_ma
	1916119.0   gstlalcbc       9/9  07:47   0+02:50:08 R  0   0.0  gstlal_ll_inspiral
	1916120.0   gstlalcbc       9/9  07:47   0+02:50:08 R  0   0.0  gstlal_ll_inspiral
	1916121.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1464.8 gstlal_inspiral --
	1916122.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1220.7 gstlal_inspiral --
	1916123.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   2197.3 gstlal_inspiral --
	1916124.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1953.1 gstlal_inspiral --
	1916125.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   2197.3 gstlal_inspiral --
	1916126.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1709.0 gstlal_inspiral --
	1916127.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   2197.3 gstlal_inspiral --
	1916128.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1709.0 gstlal_inspiral --
	1916129.0   gstlalcbc       9/9  07:47   0+02:49:50 R  0   1953.1 gstlal_inspiral --
	1916130.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   1709.0 gstlal_inspiral --
	1916131.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   1953.1 gstlal_inspiral --
	1916132.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   1464.8 gstlal_inspiral --
	1916133.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   1953.1 gstlal_inspiral --
	1916134.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   1953.1 gstlal_inspiral --
	1916135.0   gstlalcbc       9/9  07:47   0+02:49:49 R  0   2197.3 gstlal_inspiral --
	1916137.0   gstlalcbc       9/9  07:47   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916138.0   gstlalcbc       9/9  07:47   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916139.0   gstlalcbc       9/9  07:47   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916140.0   gstlalcbc       9/9  07:47   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916141.0   gstlalcbc       9/9  07:47   0+02:49:26 R  0   1709.0 gstlal_inspiral --
	1916143.0   gstlalcbc       9/9  07:48   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916144.0   gstlalcbc       9/9  07:48   0+02:49:26 R  0   1953.1 gstlal_inspiral --
	1916145.0   gstlalcbc       9/9  07:48   0+02:49:26 R  0   2441.4 gstlal_inspiral --
	1916146.0   gstlalcbc       9/9  07:48   0+02:49:25 R  0   1953.1 gstlal_inspiral --
	1916147.0   gstlalcbc       9/9  07:48   0+02:49:38 R  0   0.0  lvalert_listen --u



\section Review  Review



\subsection day1  July 29, 2015


In attendance: Chad, Edward Maros, Gareth Tomas, Jolien Creighton, Kent Blackburn, Laleh Sadeghian, Les Wade, Maddie Wade, Sathyaprakash, Sarah Caudill, Tjonnie Li

\subsubsection day1news News

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
see if the situation improves as we go forward.  For more information on our
KDE see: https://dcc.ligo.org/G1500589

\subsubsection day1completed Completed Actions

 - fix the missed found plot in the iHope pages: <https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/online-test/Images/H1L1V1-GSTLAL_INSPIRAL_PLOTSUMMARY_ALL_LLOID_COMBINED_01_deff_vs_t_H1L1_closedbox-1121652724-550359.png>. Done, see: 1ea0df6af5ec219db95c0784ca186e40acfb91d0
 
 - Remove candidate event(s) from closed box results. Done see: a886f41349ee2369615cb49455dfcfb7695bcc7a

 - Make the information about injection rate available on the summary pages along side the parameters used for injections: Done see above
 
 - Instead of overwriting online iHope pages it would be good to save the old one (with a time stamp) and start afresh but with all the data so that we can have the history available to look at. Done, see: 43cc68e42c882a4710196ebe51633432b0ad2bca

 - Use a smoothing kernel with a larger (try 2, 4) scale to smoothen the likelihood and probability plots. This has been investigated and is documented here: https://bugs.ligo.org/redmine/issues/2339


\subsection day2  July 30, 2015


In attendance: Chad, Jolien and Sathya

\subsubsection day2news News

 - *Big Dog was detected the night before*

\subsubsection day2actions Actions

 - Look at missed close-by injections (there are not many but it is puzzling why ~8 Mpc injections are missed; could be due to injections being on lock-loss boundary but should be investigated).

 - Use integer GPS second for the on-line clustering script.

 - The SNR heat plot and the SNR-Chisquare trigger plots do not appear in SIM-DB page; it would be good to have them there too to compare with the Big Dog event and to see if the anatomy of the triggers is similar when there is a loud event. 

\subsubsection day2completed Completed Actions


\subsection day3 July 31, 2015


In attendance: Chad, Sathya

from Sathya:

 - We discussed the result of applying a kernel that uses a larger scale and you
found that using a uniformly large scale degrades the sensitivity even in
regions where you already have enough triggers to measure the ranking statistic
accurately. Showed how a trigger density-dependent (I think you called
it dynamic) scaling would work and I am happy with the outcome.

 - I would like to invite the other reviewers to look the results of applying the two different kernels and see if they are happy. We should review your patch to make sure there is nothing obviously wrong and we can do that when there are at least two reviewers present.

 - You also pointed out that the online webpage IFAR plots were actually still "biased" with respect to the prediction (as opposed to what you erroneously showed us yesterday, which seemed to show them to have converged); but it is now less biased than two days ago, hopefully the measured IFAR will converge to the predicted one soon.

\subsubsection day3news News

\subsubsection day3actions Actions

 - The only action item is for the BBH group to use your patch for the ranking statistic and rerun to see if the new kernel causes any problem. We both agreed that you would need to test the kernel for a while before pushing the change to O1 release. 

 - There is an action item also on me to come up with a checklist for us to go through everyday during an online run. I will circulate this before our call tomorrow (BTW when do we meet tomorrow, 10 am Eastern).

\subsubsection day3completed Completed Actions


\subsection day4 August 1, 2015


\subsubsection day4news News

Unfortunately there was a simdb outage causing several injection jobs to fail and currently they are in a flopping state awaiting response from the simdb team.  An open question is, should we trap failures such as these?

	Traceback (most recent call last):
	.............................................................
	  File "/usr/lib/python2.7/ssl.py", line 305, in do_handshake
	    self._sslobj.do_handshake()
	socket.error: [Errno 104] Connection reset by peer

\subsection day6 August 3, 2015


\subsubsection day6news News

 - The simdb outage is over, however there was a substantial downtime for
injection jobs over the weekend.  Fortunately the noninjection jobs were not
effected. The outage effected:
   - July 31st 20:51:08 CDT (First 500 response.)
   - August 1st 16:24:08 CDT (First 200 response.)

 - GraceDB uploads have nothing unusual, but we now see one 1 in a million
   second event (after roughly 5 day of the run), consistent with what should
be expected after this many days into the run. Here are somethings to think
about:

\subsubsection day6actions Actions

 - The IFAR plot has still not converged. It might be a good idea to look at
   plots from day to day to figure out the convergence. A similar issue has
been seen in IMBH search; a smaller bias but for the three detector case. The
issue here might be different.

 - There have been some SimDB failures; this might cause job crashes when the
   job runs out of 'max retry's. NOTE perhaps we should disable uploads of
   ranking data. I think that is what filled up the disk.

\subsubsection day6completed Completed Actions

 - Inform the rest of the CBC about Replay. Include replay page, GraceDB page,
   online status page, online ihope page, offline status page. Added to 8/4 CBC call.


\subsection day7 August 4, 2015


Do we want to consider a cold restart with some action items addressed?
\subsubsection day7news News

\subsubsection day7actions Actions

\subsubsection day7completed Completed Actions
 
 - Get the range vs time (realtime) plot working
 - Disable ranking data uploads to relieve stress on simdb
 - Start with a less aggressive prior to see if FAR converges more quickly.



\subsection day8 August 19, 2015

\subsubsection day8news News

After a hiatus, we decided to take some steps to try to get the FAR calculation to converge faster.  As a reminder of hte situation

 - The FARs are being over reported (i.g., events are ranked less significantly than they should be) by about a factor of 4.

 - This has very little effect on sensitivity. We recover the big dog fine.  The software injection recovery seems reasonable.   

 - However we might produce fewer events in gracedb than we want.

Our hythosis is this:  We used an agressive prior to seed the analysis to prevent significance from being *over* estimated - i.e., we were trying to be conservative. We think that this was too agressive.  Since then we have taken a step to try to tone down the prior 

<a href=https://ligo-vcs.phys.uwm.edu/cgit/gstlal/commit/gstlal-inspiral/bin/gstlal_ll_inspiral_create_prior_diststats?id=d94163250b45e1f544a6a8cf754dd4bb9a7a2e02>d94163250b45e1f544a6a8cf754dd4bb9a7a2e02</a>

We took the liberty to also try out the proposed kde patch described here:

https://bugs.ligo.org/redmine/issues/2339

The analysis has been running for less than one day.  The IFAR plot has not yet converged, but it is still too early to decide if it is working.

Additionally, Cody is rerunning the offline analysis with the new KDE patch applied.

\subsubsection day8actions Actions



\subsection day9 September 9, 2015

\subsubsection day9news News

The following has been done to improve the convergence of FAR estimation on line

 - We have reduced the prior count for the SNR/chisq distribution at startup.  This means that data starts to dominate sooner
 - We have found an fixed a bug in how zero lag events were being counted and histogramed. This has been fixed in the gstlal-inspiral release 0.6.4
 - We have lowered the wait time from 4 hours to 15 minutes at the start of the online DAG in order to have the convergence time improve.

\subsubsection day9actions Actions


\subsubsection day9completed Completed Actions



\subsection day10 September 10, 2015


\subsubsection day10news News

 - We have updated a script to reset the likelihood distribution after a cold
   start. It addresses the fact that when we start from scratch the likelihood
ratio does not have real data to work with and it assigns incorrect likelihood
values to events for the first ~1 day.  We need to be able to purge those
events to reset the FAR/FAP estimation.

 - It is now part of our plan going forward to always start an online analysis with a ~1day run and then a reboot any time there is a significant change to the template bank.

 - FTR, the first "burn in" phase started on: ~1125838300
 - We reset the likelihood after a few hours at: 1125877400 and let it run again to see if even with very little data we could e.g., detect the big dog.  This was detected quite reliablly as T182998.
 - We reset the likelihood again at 1125952213 (after about 1 day) and are now settled into a long run.

\subsubsection day9actions Actions

\subsubsection day9completed Completed Actions
