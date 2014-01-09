\page low_latency_search_doc Low-latency, online search documentation

\section Introduction Introduction
 
- start by making a directory where you will run the analysis:

		$ mkdir /home/gstlalcbc/engineering/5

Gotchas:

- You will need a robot certificate in order to communicate with gracedb and other services requiring authentication. https://wiki.ligo.org/AuthProject/LIGOCARobotCertificate 
- You will need your own lv alert account https://www.lsc-group.phys.uwm.edu/daswg/docs/howto/lvalert-howto.html

\section Banks Preparing the template banks

- First make a directory for the template banks, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/bns_bank

- Next obtain a Makefile to automate the bank generation, e.g. : <a href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERBank>this example</a>

		$ wget http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERBank

- Then modify the make file to suit your needs, e.g. changing the masses or low frequency starting points

- The run make

		$ make -f Makefile.ERBank

- After several minutes this will result in a dag file that can be submited by doing

		$ condor_submit_dag bank.dag

- You can track the progress of the dag by doing:

		$ tail -f bank.dag.dagman.out

\section Analysis Setting up the analyis dag

- begin by making a directory for the analysis dag to run, e.g.,

		$ mkdir /home/gstlalcbc/engineering/5/bns_bank

- next obtain a makefile to automate the dag generation, e.g., <a href=http://www.lsc-group.phys.uwm.edu/cgit/gstlal/plain/gstlal-inspiral/share/Makefile.ERTrigs>this example</a>

- Modify the makefile to your liking and then run make seed

		$ make seed -f Makefile.ERTrigs

- make seed is the first and necessary step when starting a new analysis from scratch. It sets up some of the necessary input files and produces a dag that will *not* submit events to gracedb.  This is important because the online analysis needs time to run and collect background statistics.

- Once make is finished condor submit the dag

		$ condor_submit_dag trigger_pipe.dag

- At this point one must wait until a sufficient sample of background is collected (at least 24 hours of triple coincident time).  While the dag is running you can monitor it by going to the monitoring section of this document

- Once the dag has run for a sufficient time to collect background statistics condor remove it and wait for the jobs to finish.  **NOTE: Since these online jobs run "forever man" they rely on condor sending them a soft kill (sig 15) rather than a hard kill (sig 9).  The jobs intercept signal 15 and perform some cleanup steps to shutdown the analysis and write out the final output files.  This is a necessary step, otherwise data will be lost.**

- Next you have to remake the dag, but without the "seed" configuration.

		$ make -f Makefile.ERTrigs 

- this will overwrite your dag file, but not other files, like logs, so you will need to force resubmission

		$ condor_submit_dag -f trigger_pipe.dag

- Now gracedb event uploading will be enbabled and the analysis is in production mode

\subsection far_thresh Adjusting the gracedb FAR threshold

Each gstlal_inspiral job that is running is also running its own webserver as a way to request information about the job or to post new configuration information to the job.  One very useful result of this is a way to dynamically change the FAR threshold used to submit gracedb events.  This can be done from the triggers directory with

		$ gstlal_ll_inspiral_gracedb_threshold --gracedb-far-threshold <FAR THRESH> *registry.txt

\subsection setting up an lvalert_listen process to automate some event followup

		
\section monitor Monitoring the output

