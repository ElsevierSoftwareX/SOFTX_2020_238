\page gstlalinspirallowlatencysearchpage Low-latency, online search documentation

[TOC]

\section Introduction Introduction
 
- start by making a directory where you will run the analysis:

		$ mkdir /home/gstlalcbc/engineering/5

\subsection Gotchas Gotchas:

- You will need a robot certificate in order to communicate with gracedb and
  other services requiring authentication.
https://wiki.ligo.org/AuthProject/LIGOCARobotCertificate 

- You will need your own lv alert account
  https://www.lsc-group.phys.uwm.edu/daswg/docs/howto/lvalert-howto.html

\section Banks Preparing the template banks

- First make a directory for the template banks, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/bns_bank

- Next obtain a Makefile to automate the bank generation, e.g. : <a
  href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERBank>this
example</a>

		$ wget http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERBank

- Then modify the make file to suit your needs, e.g. changing the masses or low
  frequency starting points

- The run make

		$ make -f Makefile.ERBank

- After several minutes this will result in a dag file that can be submited by
  doing

		$ condor_submit_dag bank.dag

- You can track the progress of the dag by doing:

		$ tail -f bank.dag.dagman.out

\subsection Resources Resources used 

- gstlal_fake_frames
- lalapps_tmpltbank
- gstlal_bank_splitter
- gstlal_psd_xml_from_asd_txt
- ligolw_add
- lalapps_path2cache
- gstlal_inspiral_svd_bank_pipe


\section Analysis Setting up the analyis dag

- begin by making a directory for the analysis dag to run, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/bns_bank

- next obtain a makefile to automate the dag generation, e.g., <a
  href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERTrigs>this
example</a>

- Modify the makefile to your liking and then run make seed

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

- make seed is the first and necessary step when starting a new analysis from
  scratch. It sets up some of the necessary input files and produces a dag that
will *not* submit events to gracedb.  This is important because the online
analysis needs time to run and collect background statistics.

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

- this will overwrite your dag file, but not other files, like logs, so you
  will need to force resubmission

		$ condor_submit_dag -f trigger_pipe.dag

- Now gracedb event uploading will be enbabled and the analysis is in
  production mode

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
