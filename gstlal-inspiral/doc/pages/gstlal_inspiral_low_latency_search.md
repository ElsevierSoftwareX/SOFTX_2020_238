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
some of the Universe's most violent transient astronomical events - such as
gamma-ray bursts - and set the stage to discover completely unanticipated
phenomena.

\subsection Algoritm Algorithm development and search pipeline

Matched filtering is the baseline method to detect compact binaries. It is
typically implemented in the frequency domain using fast Fourier transforms
that are several times longer than the duration of the underlying signal (<a href=http://arxiv.org/abs/gr-qc/0509116>Phys. Rev. D 85, 122006 (2012)</a>).  Binary
neutron stars may be observable for more than 30 minutes in the advanced LIGO
band, which implies that the standard matched filtering paradigm incurs a  O(1)
hr latency.  Naively implementing a time-domain matched filtering algorithm
through brute force convolution would achieve the latency goals, but it would
require O(10) GFLOPS per gravitational wave template
(<a href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>), which is not
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

		$ mkdir /home/gstlalcbc/engineering/5

\subsection Gotchas Gotchas:

- You will need a robot certificate in order to communicate with gracedb and
  other services requiring authentication.
https://wiki.ligo.org/AuthProject/LIGOCARobotCertificate 

- You will need your own lv alert account
  https://www.lsc-group.phys.uwm.edu/daswg/docs/howto/lvalert-howto.html

\section Banks Preparing the template banks

\subsection banks_description Description of the workflow

The following graph shows the sequence of events to make the online template
banks. These steps are mostly automated by a Makefile and a condor dag.

Steps in the Makefile

@dotfile bankgeneration.dot

\subsection Instructions

- First make a directory for the template banks, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/bns_bank

- Next obtain a Makefile to automate the bank generation, e.g. : <a
  href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.BNSERBank>this
example</a>

		$ wget http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.BNSERBank

- Then modify the make file to suit your needs, e.g. changing the masses or low
  frequency starting points

- Then run make

		$ make -f Makefile.BNSERBank

- After several minutes this will result in a dag file that can be submited by
		$ condor_submit_dag bank.dag

- You can track the progress of the dag by doing:

		$ tail -f bank.dag.dagman.out

\subsection Resources Resources used 

- gstlal_fake_frames
- lalapps_tmpltbank
- gstlal_bank_splitter
- gstlal_psd_xml_from_asd_txt
- ligolw_add
- gstlal_inspiral_svd_bank_pipe

\section Analysis Setting up the analysis dag

A makefile automates the construction of a HTCondor DAG.  The steps in the
makefile are shown in the diagram below.  The dag requires the template banks
set up in the previous section.

@dotfile analysisgeneration.dot

- Begin by making a directory for the analysis dag to run, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/analysis

- Next obtain a makefile to automate the dag generation, e.g., <a
  href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERTrigs>this
example</a>

- Modify the makefile to your liking (make sure it knows where the files you made with the bank dag are) and then run make seed

		$ make seed -f Makefile.ERTrigs

	Note that you will be prompted during the dag creation stage for your
lvalert username and password.  The password for lvalert is **not** secure. It
will show up in plain text on the submit node.  You should not use any password
that is used elsewhere (like your ligo.org password)  Since this dag is for
running on LDG resources, the plain text should not be much of a problem.  One
should not check in any code or makefiles that contain this information (hence
why you are asked for it).  lvalert is a thin layer only sending announcments.
gracedb, where the real data is stored, still requires proper ligo
authentication.

- Take note of the last few lines of the dag generation step, it will provide a
  url where you can monitor the output (described below).  It should look
something like this:

		NOTE! You can monitor the analysis at this url: https://ldas-jobs.ligo.caltech.edu/~gstlalcbc/cgi-bin/gstlal_llcbcsummary?id=0001,0009&dir=/mnt/qfs3/gstlalcbc/engineering/5/bns_trigs_40Hz

- Make seed is the first and necessary step when starting a new analysis from
  scratch. It sets up some of the necessary input files and produces a dag that
will *not* submit events to gracedb.  This is important because the online
analysis needs time to run and collect background statistics.

\section Running Running the analysis dag

- Once make is finished condor submit the dag

		$ condor_submit_dag trigger_pipe.dag

- At this point one must wait until a sufficient sample of background is
  collected (at least 24 hours of triple coincident time).  While the dag is
running you can monitor it by going to the monitoring section of this document

- Once the dag has run for a sufficient time to collect background statistics
  condor remove it and wait for the jobs to finish.  **NOTE: Since these online
jobs run "forever man" they rely on condor sending them a soft kill (sig 15)
rather than a hard kill (sig 9).  The jobs intercept signal 15 and perform some
cleanup steps to shutdown the analysis and write out the final output files.
This is a necessary step, otherwise data will be lost.**

- Next you have to remake the dag, but without the "seed" configuration.

		$ make -f Makefile.ERTrigs 

- This will overwrite your dag file, but not other files, like logs, so you
  will need to force resubmission

		$ condor_submit_dag -f trigger_pipe.dag

- Now gracedb event uploading will be enbabled and the analysis is in
  production mode

The running dag topology looks like this:

@dotfile llpipe.dot

\subsection far_thresh Adjusting the gracedb FAR threshold

Each gstlal_inspiral job that is running is also running its own webserver as a
way to request information about the job or to post new configuration
information to the job.  One very useful result of this is a way to dynamically
change the FAR threshold used to submit gracedb events.  This can be done from
the triggers directory with

		$ gstlal_ll_inspiral_gracedb_threshold --gracedb-far-threshold <FAR THRESH> *registry.txt

\subsection Resources Resources used

- gstlal_ll_trigger_pipe
- gstlal_inspiral_reset_likelihood
- gstlal_ll_inspiral_gracedb_threshold
- gstlal_inspiral_create_prior_diststats
- gstlal_inspiral_marginalize_likelihood

		
\section monitor Monitoring the output

As mentioned above you can monitor the output.  Please see the
gstlal_llcbcsummary for more information. 

Events are uploaded to https://gracedb.ligo.org

\section Review Review status

\subsection Team Review Team 2014

- Reviewees: Chad, Kipp
- Reviewers: Jolien, Florent, Duncan Me, Sathya

\subsection pythontable Python programs and modules

Redundant entries are ommitted

<table>
<tr><th> Program				</th><th> Sub programs or modules	</th><th> Lines	</th><th> Review status	</th><th> Stability </th></tr>
<tr><td> gstlal_fake_frames			</td><td>				</td><td> 360	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pipeparts.py			</td><td> 965	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> reference_psd.py		</td><td> 648	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> simplehandler.py		</td><td> 143	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> datasource.py			</td><td> 749	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> multirate_datasource.py	</td><td> 291	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> glue.segments			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> glue.ligolw*			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.datatypes		</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.series			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> lalapps_tmpltbank			</td><td>                               </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_bank_splitter			</td><td>                               </td><td> 187	</td><td> \reviewed with actions	</td><td> \moddev </td></tr>
<tr><td>					</td><td> pylal.spawaveform		</td><td> 1244	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> glue.lal			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> templates.py			</td><td> 299	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> inspiral_pipe.py		</td><td> 279	</td><td> \reviewed with actions	</td><td> \moddev </td></tr>
<tr><td> gstlal_psd_xml_from_asd_txt		</td><td>                               </td><td> 81	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.xlal*			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> ligolw_add				</td><td>                               </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_svd_bank_pipe		</td><td>                               </td><td> 201	</td><td> \reviewed	</td><td> \moddev </td></tr>
<tr><td> 					</td><td> glue.iterutils                </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> 					</td><td> glue.pipeliene                </td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_svd_bank			</td><td>                               </td><td> 164	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> svd_bank.py			</td><td> 363	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> cbc_template_fir.py		</td><td> 443	</td><td> \notreviewed with actions	</td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_create_prior_diststats	</td><td>                               </td><td> 125	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td>					</td><td> far.py			</td><td> 1714	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td>					</td><td> pylal.inject			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.rate			</td><td> NA	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.snglcoinc		</td><td> ?	</td><td> ?		</td><td> \moddev </td></tr>
<tr><td> gstlal_inspiral_marginalize_likelihood	</td><td>                               </td><td> 167	</td><td> \notreviewed	</td><td> \moddev </td></tr>
<tr><td> gstlal_ll_trigger_pipe			</td><td>                               </td><td> -	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td> gstlal_inspiral			</td><td>                               </td><td> 707	</td><td> \reviewed with actions	</td><td> \moddev </td></tr>
<tr><td>					</td><td> lloidparts.py			</td><td> 826	</td><td> \reviewed with actions	</td><td> \stable </td></tr>
<tr><td>					</td><td> pipeio.py			</td><td> 239	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> simulation.py			</td><td> 72	</td><td> \reviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> inspiral.py			</td><td> 949	</td><td> \notreviewed	</td><td> \moddev </td></tr>
<tr><td>					</td><td> streamthinca.py		</td><td> 387	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td>					</td><td> pylal.ligolw_thinca		</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td>					</td><td> httpinterface.py		</td><td> 110	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td>					</td><td> hoftcache.py			</td><td> 110	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_llcbcsummary			</td><td>                               </td><td> 450	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_llcbcnode			</td><td>                               </td><td> 318	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_lvalert_psd_plotter	</td><td>                               </td><td> 240	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_ll_inspiral_get_urls		</td><td>                               </td><td> 93	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_followups_from_gracedb	</td><td>                               </td><td> 177	</td><td> \notreviewed	</td><td> \stable </td></tr>
<tr><td> gstlal_inspiral_recompute_online_far_from_gracedb </td><td>                    </td><td> 18	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td> gstlal_inspiral_recompute_online_far   </td><td>                    		</td><td> 92	</td><td> \notreviewed	</td><td> \hidev </td></tr>
<tr><td> gstlal_inspiral_calc_likelihood   	</td><td>                    		</td><td> 409	</td><td> \notreviewed	</td><td> \moddev </td></tr>
<tr><td>					</td><td> pylal.burca2			</td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
<tr><td> gstlal_ll_inspiral_gracedb_threshold	</td><td>                               </td><td> 106	</td><td> \notreviewed	</td><td> \moddev </td></tr>
<tr><td> lvalert_listen				</td><td>                               </td><td> ?	</td><td> ?		</td><td> \stable </td></tr>
</table>


\subsection gsttable gstreamer elements

<table>
<tr><th> Element					</th><th> depenedencies		</th><th> # lines </th><th> Review status	</th><th> Stability	</th></tr>
<tr><td> \ref pipeparts.mkwhiten() lal_whiten		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mktogglecomplex() lal_togglecomplex</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mksumsquares() lal_sumsquares	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkstatevector() lal_statevector	</td><td>			</td><td> 	  </td><td> \reviewed 		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkinjections() lal_simulation 	</td><td>			</td><td> 	  </td><td> \notreviewed (with actions)	</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mksegmentsrc() lal_segmentsrc 	</td><td>			</td><td> 	  </td><td> \reviewed (with actions)	</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkreblock() lal_reblock 	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkpeak() lal_peak		</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mknxydump() lal_nxydump		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mknofakedisconts() lal_nofakedisconts</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkmatrixmixer() lal_matrixmixer	</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable	</td></tr>
<tr><td> \ref pipeparts.mkgate() lal_gate		</td><td>			</td><td> 	  </td><td> \reviewed (with actions)	</td><td> \stable       </td></tr>
<tr><td> \ref pipeparts.mkfirbank() lal_firbank		</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkdrop() lal_drop		</td><td>			</td><td> 	  </td><td> \reviewed (with actions)	</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkcachesrc() lal_cachesrc	</td><td>			</td><td> 	  </td><td> \reviewed (with actions)	</td><td> \stable 	</td></tr>
<tr><td> \ref pipeparts.mkitac() lal_itac		</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \moddev	</td></tr>
<tr><td> framecpp_filesink				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable 	</td></tr>
<tr><td> framecpp_channelmux				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable 	</td></tr>
<tr><td> framecpp_channeldemux				</td><td>			</td><td> 	  </td><td> \notreviewed	</td><td> \stable	</td></tr>
<tr><td> gds_framexmitsrc				</td><td>			</td><td> 	  </td><td> \reviewed		</td><td> \stable 	</td></tr>
<tr><td> gds_lvshmsrc					</td><td>			</td><td> 	  </td><td> \reviewed (with actions)	</td><td> \stable 	</td></tr>
</table>

\subsection Broader action items

- Test robustness of fixed bank (start by figuring out the right question!)
 - Sathya to contact Marcel Kehl to enquire about goals of testing constant template banks.
- Analysis Makefiles should be documented (e.g., parameters); Do we want them to be made more generic?
 - *Chad: This process has been started.  See, e.g., Makefile.triggers_example*
- Test delta function input to LLOID algorithm (e.g with and without SVD)
- Consider how to let the user change SNR threshold consistently (if at all).  Note this is tied to SNR bins in far.py
- Add synopses for all programs in documentation
 - *Chad: Done*
- Background estimations should have more informative plots e.g., smoothed likelihood functions
- Study the dependence of coincidence triggers on SNR threshold
- Document offline pipeline including graphs of workflows
 - *Chad: Done*, see \ref gstlalinspiralofflinesearchpage
- Test pipeline with control peak times set to different values
 - *Chad: Done* see \ref gstlalinspiralcontrolpeaktimestudy
- Write documentation for autochisq (paper in progress)
- Write joint likelihood ranking and FAP calculation (paper in progress)
- Explore autocorrelation chisquared mismatch scaling with number of samples e.g., @f$ \nu + \epsilon(\nu) \delta^{2} @f$
- Make sure capsfilters and other similar elements are well documented within graphs (e.g. put in rates, etc)
- Add description of arrows in graphs
- Verify that all the segments are being tracked in online mode via the Handler (this is coupled to the inspiral.Data class, so it will come up again there)
- Feature request for detchar - It would be helpful to have online instrument state that could be queried to know if an instrument will be down for an extended time

![Kipp explaining that the Flux Capacitor is what makes time travel possible; Jolien responds incredulously at the one point twenty-one jiga-flops of power required](@ref kipp.png)  ![LLOID](@ref lloid.png) 
