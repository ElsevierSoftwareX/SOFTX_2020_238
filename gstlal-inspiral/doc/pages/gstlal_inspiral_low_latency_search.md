\page gstlalinspirallowlatencysearchpage Low-latency, online search documentation

[TOC]

\section Introduction Introduction

The prospect of detecting tens of compact binary mergers per year with Advanced
LIGO and Virgo is ushering in a new era of gravitational-wave astronomy with
potential for joint electromagnetic and astroparticle observations.  The
Compact Binary Coalescence group in the LIGO Scientific Collaboration (LSC) has
identified the low-latency detection of gravitational waves, i.e., detection in
less than 1 minute from the signal arriving at Earth, from coalescing neutron
star and black hole binary systems as a science priority. Joint gravitational
wave and electromagnetic observations will determine the progenitor models of
some of violent transient astronomical events - such as
gamma-ray bursts - and set the stage to discover completely unanticipated
phenomena.

\subsection Algoritm Algorithm development and search pipeline

Matched filtering is the baseline method to detect compact binaries. It is
typically implemented in the frequency domain using fast Fourier transforms
that are several times longer than the duration of the underlying signal (<a
href=http://arxiv.org/abs/gr-qc/0509116>Phys. Rev. D 85, 122006 (2012)</a>).
Binary neutron stars may be observable for more than 30 minutes in the advanced
LIGO band, which implies that the standard matched filtering paradigm incurs a
O(1) hr latency.  Naively implementing a time-domain matched filtering
algorithm through brute force convolution would achieve the latency goals, but
it would require O(10) GFLOPS per gravitational wave template (<a
href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>), which is not
feasible.

\subsubsection LLOID The Low Latency Online Inspiral Detection (LLOID) algorithm

Significant algorithmic development has led to computationally viable
low-latency methods to implement matched filter searches for compact binaries
(<a href=http://arxiv.org/abs/1112.6005>Astron Astrophys 541 A155 (2012)</a>,
<a href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>,
<a href=http://arxiv.org/abs/1108.3186>Phys Rev D 86 024012 (2012)</a>) by compressing
the waveform parameter and/or sampling space. The results are algorithms with
similar computational cost to the traditional FFT based search but with much
lower latency.

The LLOID algorithm described in <a href=http://arxiv.org/abs/1107.2665>ApJ 748 136
(2012)</a> applies two techniques to significantly reduce the computational cost
of time-domain filtering.

-# Singular value decomposition (SVD) of template waveforms
(<a href=http://arxiv.org/abs/1005.0012>Phys Rev D82 044025 (2010)</a>) to exploit
degeneracy in the compact binary parameter space and avoid performing
unnecessary template waveform convolutions.

-# Exploiting the chirp (increasing frequency and amplitude) time-frequency
evolution of compact binary systems to critically sample the templates
according to the instantaneous Nyquist sampling limit by decomposing the strain
data and templates into multiple power-of-two sample rates (multirate
filtering).

The use of SVD and multirate filtering retains 99.7% of the signal-to-noise
ratio (loses only 0.3%) obtained from conventional matched filtering. This can
be compared to the acceptable signal-to-noise ratio loss from the discreteness
of template banks, i.e., 3% (<a href=http://arxiv.org/abs/gr-qc/9808076>Phys. Rev. D
60, 022002 1999</a>).  The SVD and multirate filtering paradigm therefore leads
to at most a 1% loss in event rate over a conventional matched filter search.

The LLOID algorithm applied to advanced LIGO BNS searches estimates the
filtering cost for a real-time compact binary search to be O(1) MFLOPS per
gravitational wave template for the advanced LIGO design sensitivity using the
waveform decomposition described in <a href=http://arxiv.org/abs/1107.2665>ApJ 748
136 (2012)</a>.  It is an open question, however, if different choices of
template decomposition combined, perhaps, with special hardware features could
further reduce the computation cost.  Since that is not currently known the
remaining estimates on this page use the decomposition described in
<a href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>.  

The O(1) MFLOPs cost neglects all other costs of performing the analysis
including additional signal consistency checks, coincidence analysis,
background estimation, etc., but it roughly sets the order of magnitude
required to do a low latency search with the LLOID algorithm.

\subsection literature Relationship to published work 

- <a href=http://arxiv.org/abs/1005.0012>Phys Rev D82 044025 (2010)</a>:
  Provides basic expressions for computing the singular value decomposition of
waveforms and expectations for signal-to-noise ratio loss, etc.
- <a href=http://arxiv.org/abs/1101.0584> Phys Rev D83 084053 (2011)</a>
  describes the composite detection statistic that is formed from the singular
value decomposition basis vectors.  This statistic allows one to infer the
presence of signals in the data without reconstructing the physical SNRs.  This
is used optionally in gstlal inspiral pipelines.  Using it saves substantial
CPU cost but at the cost of some detection efficiency.  Unlike the published
result, we do not threshold on the composite detection statistic value, rather
we do "peak finding" with the composite detection statistic time series and
reconstruct physical SNRs around the peaks.  The peak time window is a free
parameter.
- <a href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>: The so called
  "LLOID" algorithm combines singular value decomposition with multirate
sampling of the data and waveforms to produce a computationally efficient
time-domain algorithm.  The algorithm described in the paper is basically
identical to what is done.
- <a href=http://arxiv.org/abs/1209.0718>Phys Rev D88 024025 (2013)</a> The
  basic principle of significance estimation in the gstlal inspiral pipeline is
described in this paper.  Since this paper was published, the likelihood ratio
ranking statistic has been extended to more accuratly account for correlated
(between detector) signal probabilities, which was ignored in the published
work.  This has led to substantial sensitivity improvement, but also a more
computationally complex background estimation procedure which is still being
developed.  The first 5 engineering runs used the simpler procedure as
published.  The 6th engineering run will be the first to use the new procedure. 

\section Preliminaries Preliminaries

- Start by making a directory where you will run the analysis:

		$ mkdir /home/gstlalcbc/observing/1

\subsection Gotchas Gotchas for automated services, e.g., gracedb submission, ligoDV web, etc.:

- You will need a robot certificate in order to communicate with gracedb and
  other services requiring authentication.
https://wiki.ligo.org/AuthProject/LIGOCARobotCertificate 

- You will need your own lv alert account
  https://www.lsc-group.phys.uwm.edu/daswg/docs/howto/lvalert-howto.html

- You will need a kerberos keytab and to point the KERBEROS_KEYTAB environment
  variable to it: https://wiki.ligo.org/AuthProject/CommandLineLIGOAuth

\section Banks Preparing the template banks

\subsection Instructions

- First make a directory for the template banks, e.g.,

		$ mkdir /home/gstlalcbc/observing/1/svd_bank

- Next obtain a Makefile to automate the bank generation, e.g. : <a href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.online_bank>this
example</a>

		$ wget http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.online_bank

- Then modify the make file to suit your needs, e.g. changing the masses or low
  frequency starting points

- Then run make

		$ make -f Makefile.online_bank

- After several minutes this will result in a dag file that can be submited by
		$ condor_submit_dag bank.dag

- You can track the progress of the dag by doing:

		$ tail -f bank.dag.dagman.out

\subsection Resources Resources used 

- gstlal_bank_splitter
- gstlal_inspiral_svd_bank_pipe

\section Analysis Setting up the analysis dag

A makefile automates the construction of a HTCondor DAG.  The dag requires the
template banks set up in the previous section.

**NOTE: There is a burn-in phase to online analysis.  You will need to run for
a day or two (at least) to gather background statistics, reset some components,
and start a new analysis derived from the burn-in results.**

- Begin by making a directory for the analysis dag to run, e.g.,

		$ mkdir /home/gstlalcbc/observing/1/trigs.burnin

- Next obtain a makefile to automate the dag generation, e.g., <a
  href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.online_analysis>this
example</a>

- Modify the makefile to your liking. **NOTE:*It is critical that you disable
  uploads to gracedb and simdb when doing this burn-in phase.  You must set:** 

		--gracedb-far-threshold -1
		--inj-gracedb-far-threshold -1

- Then you can run make

		$ make Makefile.online_analysis

- Note that you will be prompted during the dag creation stage for your lvalert
username and password.  The password for lvalert is **not** secure. It will
show up in plain text on the submit node.  You should not use any password that
is used elsewhere (like your ligo.org password)  Since this dag is for running
on LDG resources, the plain text should not be much of a problem.  One should
not check in any code or makefiles that contain this information (hence why you
are asked for it).  lvalert is a thin layer only sending announcments.
gracedb, where the real data is stored, still requires proper ligo
authentication.

- Take note of the last few lines of the dag generation step, it will provide a
  url where you can monitor the output (described below).  It should look
something like this:

		NOTE! You can monitor the analysis at this url: https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary?id=0001,0009&dir=/mnt/qfs3/gstlalcbc/observing/1/trigs&ifos=H1,L1

\section Running Running the analysis dag in burn-in phase

- Once make is finished condor submit the dag

		$ condor_submit_dag trigger_pipe.dag

- Once the dag has run for a sufficient time to collect background statistics
  condor remove it and wait for the jobs to finish.  **NOTE: Since these online
jobs run "forever man" they rely on condor sending them a soft kill (sig 15)
rather than a hard kill (sig 9).  The jobs intercept signal 15 and perform some
cleanup steps to shutdown the analysis and write out the final output files.
This is a necessary step, otherwise data will be lost.**

\section Reconfiguring the dag after the burn in phase

- Make a new directory e.g, 

		$ mkdir /home/gstlalcbc/observing/1/trigs

- Copy the makefile and *prior.xml.gz from your burn-in directory

- Reset some of the likelihood data with

		$ make -f Makefile.online_analysis reset-likelihood

- Then remake the dag

		$ make -f Makefile.online_analysis

- And condor submit it

		$ condor_submit_dag -f trigger_pipe.dag

The running dag topology looks like this:

@dotfile llpipe.dot

The contents of the box labelled ``Global background statistics processing'' and a more detailed flow chart showing its relationship with the calculations performed by each candidate generating job is shown in the following diagram:

@dotfile rankingflow.dot

\subsection far_thresh Adjusting the gracedb FAR threshold

Each gstlal_inspiral job that is running is also running its own webserver as a
way to request information about the job or to post new configuration
information to the job.  One very useful result of this is a way to dynamically
change the FAR threshold used to submit gracedb events.  This can be done from
the triggers directory with

		$ gstlal_ll_inspiral_gracedb_threshold --gracedb-far-threshold <FAR THRESH> *registry.txt

\subsection Resources Resources used

- gstlal_inspiral
- gstlal_ll_inspiral_get_urls
- gstlal_ll_trigger_pipe
- gstlal_ll_inspiral_gracedb_threshold
- gstlal_inspiral_ll_create_prior_diststats
- gstlal_inspiral_reset_likelihood
- gstlal_inspiral_marginalize_likelihood
- gstlal_inspiral_marginalize_likelihoods_online

		
\section monitor Monitoring the output

Output is available on these timescales

- Seconds:
	- uploads to gracedb.ligo.org
	- uploads to simdb.cgca.uwm.edu

- Minutes:
	- Monitoring pages at e.g., https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/cgi-bin/gstlalcbcsummary
	- The corresponding "node" links at the bottom of the page above

- Hours:
	- An offline style page that summarizes the cumulative run in the directory specified in the Makefile as: 

		WEBDIR=

