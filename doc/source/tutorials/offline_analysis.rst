Running an offline compact binary coalescence analysis
========================================================================

Prerequisites
-------------

 - Fully functional gstlal, gstlal-ugly, gstlal-inspiral installation
 - Condor managed computing resource using the LIGO Data Grid configuration
 - Access to gravitational wave data, stored locally or via CVMFS

Introduction
------------

This tutorial will help you to setup and run a offline gravitational wave search for binary neutron stars. The information contained within this document can easily be modified to perform a wide range of searches.

The offline analysis has a somewhat involved setup procedure which is usually performed with use of a Makefile. This documentation covers everything needed to set up a offline search. The analysis itself is performed by a pipeline contained within a dag (Directed Acyclic Graph) that is managed by condor. The dag and job sub files are produced by running ``gstlal_inspiral_pipe``. This program requires several input files that are produced in several steps, all of which are detailed below. These input files are:

 * segments.xml.gz
 * vetoes.xml.gz
 * frame.cache
 * inj_tisi.xml
 * tisi.xml
 * injection file
 * split bank cache files

The steps to produce the full analysis dag file are:

 1. Set analysis variables defined at top of offline Makefile.
 2. Generate frame cache, segments, vetoes, and tisi files.
 3. Produce injection file.
 4. Generate/copy template bank and then split this into sub-banks.
 5. Run ``gstlal_inspiral_pipe`` to produce offline analysis dag and sub files.

The information contained within this page is based off the O2 BNS HL test dag, an offline analysis focused on 100,000s centered around GW170817. The dag used to perform the analysis can be produced using a `Makefile <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/O3/offline/O2/Makefile.BNS_HL_test_dag_O2>`_ that generates most of the required files. This tutorial will just cover the HL detector pair configuration, though a HLV Makefile can be found `here <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/O3/offline/O2/Makefile.BNS_HLV_test_dag_O2>`_. In this tutorial we detail each stage of the Makefile needed to run an offline analysis.

Analysis variables defined at the top of offline Makefile
---------------------------------------------------------

There many variables that are set at the top of the offline Makefile. Some of these should not be changed unless you know what you are doing. The variables that should be changed/set are explained here::

 ACCOUNTING_TAG=ligo.dev.o3.cbc.uber.gstlaloffline

An accounting tag used to measure LDG computational use. See https://ldas-gridmon.ligo.caltech.edu/ldg_accounting/user. ::

 GROUP_USER=albert.einstein

This should be your albert.einstein user identification. This is only needed if using a shared account. ::

 IFOS = H1 L1
 MIN_IFOS = 2

Define which detectors to include within the analysis. H1, L1, and V1 are currently supported. Set minimum number of operational detectors for which to analyse. Able to analyse single detector time. ::

 START = 1187000000
 STOP = 1187100000

Set start and stop time of the analysis in GPS seconds. The times stated here are 100,000s around GW170817. See https://www.gw-openscience.org/gps/ for GPS time conversions. ::

 TAG = BNS_test_dag
 RUN = run_1
 WEBDIR = ~/public_html/testing/$(TAG)/$(START)-$(STOP)-$(RUN)

Set output directory for summary page of results. ::

 MCHIRP_INJECTIONS := 0.5:100.0:1_injections.xml

Used to specify injection file, and chirpmass range over which to filter it. Multiple injection files can be given at once, these should be space separated, with no whitespace at the end of the line.

**NOTE, an injection file must be passed to gstlal_inspiral_pipe, it is unable to run without one.** ::

 VETODEF = /path/to/H1L1-CBC_VETO_DEFINER_CLEANED_C02_O2_1164556817-23176801.xml

Veto definer file. Used to determine which data to veto. See https://git.ligo.org/detchar/veto-definitions/tree/master/cbc for all veto definer files. ::

 # GSTLAL_SEGMENTS Options
 SEG_SERVER=https://segments.ligo.org
 # C02 cleaned
 LIGO_SEGMENTS="$*:DCH-CLEAN_SCIENCE_C02:1"

 # The LIGO frame types
 # C02 cleaned
 HANFORD_FRAME_TYPE='H1_CLEANED_HOFT_C02'
 LIVINGSTON_FRAME_TYPE='L1_CLEANED_HOFT_C02'

 # The Channel names.
 # C02 cleaned
 H1_CHANNEL=DCH-CLEAN_STRAIN_C02
 L1_CHANNEL=DCH-CLEAN_STRAIN_C02

Gravitational wave data segment, frame type, and channel name information. See https://wiki.ligo.org/LSC/JRPComm/ for full details about all observing runs. ::

 include /path/to/Makefile.offline_analysis_rules

Full path to `Makefile.offline_analysis_rules <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/Makefile.offline_analysis_rules>`_. This file contains sets of rules for string parsing/manipulation used within the main Makefile and an up-to-date version must be included.

Generate frame cache, segments, vetoes, and tisi files
------------------------------------------------------

frame.cache file
^^^^^^^^^^^^^^^^

The frame.cache file contains the full paths to the gravitational wave data .gwf files using the following format:  

Detector site identifier, frame type, start GPS time, duration, full path to file ::

 H H1__H1_CLEANED_HOFT_C02 1186998263 4096 file://localhost/hdfs/frames/O2/hoft_C02_clean/H1/H-H1_CLEANED_HOFT_C02-11869/H-H1_CLEANED_HOFT_C02-1186998263-4096.gwf

If the .gwf data files are stored locally, then you can produce individual detector frame cache files with::

 gw_data_find -o H -t $(HANFORD_FRAME_TYPE) -l -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" $*_"$$2" "$$3" "$$4" "$$5}' > H1_frame.cache
 gw_data_find -o L -t $(LIVINGSTON_FRAME_TYPE) -l -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" $*_"$$2" "$$3" "$$4" "$$5}' > L1_frame.cache

The ``awk`` command provides some formating to put the output in the required format.

If the data must be accessed via `CVMFS <https://www.gw-openscience.org/cvmfs/>`_ then the following option needs to be added to the ``gw_data_find`` arguments::

 --server datafind.ligo.org:443

And then create a combined frame.cache file with some additional formating::

 cat H1_frame.cache L1_frame.cache > frame.cache
 sed -i s/H\ $(LIGO_FRAME_TYPE)/H\ H1_$(LIGO_FRAME_TYPE)/g frame.cache
 sed -i s/L\ $(LIGO_FRAME_TYPE)/L\ L1_$(LIGO_FRAME_TYPE)/g frame.cache

segments.xml.gz and vetoes.xml.gz files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The segments.xml.gz file contains a list of all data segments that should be analysed. The vetoes.xml.gz file contains a list of all data segments that should be ignored. ::

 ligolw_segment_query_dqsegdb --segment-url=${SEG_SERVER} -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$(LIGO_SEGMENTS) --result-name=datasegments > %_segmentspadded.xml
 ligolw_no_ilwdchar $*_segmentspadded.xml

This returns an initial segments list. This command makes use of some Makefile variables to produce segmentspadded.xml files for each detector specified by $IFOS. ``ligolw_no_ilwdchar`` is run on the output files to convert some table column types from ilwd:char to int4s. This command will need to be run on any xml file produced by a non-gstlal program. ::

 ligolw_segments_from_cats_dqsegdb --segment-url=$(SEG_SERVER) --veto-file=$(VETODEF) --gps-start-time $(START) --gps-end-time $(STOP) --cumulative-categories
 ligolw_no_ilwdchar H1-VETOTIME_CAT*.xml
 ligolw_no_ilwdchar L1-VETOTIME_CAT*.xml
 gstlal_segments_operations --union --segment-name VETO_CAT1_CUMULATIVE --output-file %_CAT1_vetoes.xml --output-segment-name datasegments $*-VETOTIME_CAT1-*.xml $*-VETOTIME_CAT1-*.xml

This queries the ligo segment server for all veto types (CAT1, CAT2, and CAT3) that are defined within the veto definer file ::

 ligolw_add --output CAT1_vetoes.xml.gz $(CAT1_VETOES_FILES)
 ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time CAT1_vetoes.xml.gz
 gzip CAT1_vetoes.xml.gz

Produce CAT1 vetoes file. ::

 ligolw_add --output segdb.xml $(SEGMENTS_FILES)
 ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segdb.xml
 gstlal_segments_operations --diff --output-file segments.xml.gz segdb.xml CAT1_vetoes.xml.gz
 gstlal_segments_trim --trim $(SEGMENT_TRIM) --gps-start-time $(START) --gps-end-time $(STOP) --min-length $(SEGMENT_MIN_LENGTH) --output segments.xml.gz segments.xml.gz

Combine initial segment files with CAT1 veto times removed to produce segments.xml.gz file. ::

 ./lauras_txt_files_to_xml -i $* -c -o $*-gates.xml $*-GATES-1163203217-24537601.txt
 ligolw_no_ilwdchar $*-gates.xml
 gstlal_segments_operations --union --segment-name VETO_CAT3_CUMULATIVE --output-file %_vetoes.xml.tmp --output-segment-name vetoes $*-VETOTIME_CAT3-*.xml $*-VETOTIME_CAT3-*.xml
 gstlal_segments_operations --union --segment-name vetoes --output-file %_vetoes.xml --output-segment-name vetoes %_vetoes.xml.tmp $*-gates.xml

Include gating times into CAT3 veto times files. The gating files contain additional times to veto that are not included within the veto definer file. The ascii files are converted into readable xml files with ``lauras_txt_files_to_xml``. ::

 ligolw_add --output vetoes.xml.gz $(VETOES_FILES)
 ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time vetoes.xml.gz
 gzip vetoes.xml.gz

Combine all veto files into single vetoes.xml.gz file.

tisi.xml.gz and inj_tisi.xml.gz file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tisi (time slide) files are used for the offsetting of events used within the analysis for the calculation of the background. ::

 lalapps_gen_timeslides --instrument=H1=0:0:0 --instrument=L1=0:0:0 inj_tisi.xml

Generate injection time slides file. ::

 lalapps_gen_timeslides --instrument=H1=0:0:0 --instrument=L1=25.13274:25.13274:25.13274 bg_tisi.xml
 ligolw_add --output tisi.xml bg_tisi.xml inj_tisi.xml

Generate analysis time slides file.

Produce injection file
----------------------

As stated above, at least one injection file must be passed to ``gstlal_inspiral_pipe``. The Makefile contains a command to produce a single BNS injection set that covers the full analysis period. These parameters can be easily adjusted for different searches. Alternative injection generation codes exist, such as `lvc_rates_injections <https://git.ligo.org/RatesAndPopulations/lvc-rates-and-pop/blob/master/bin/lvc_rates_injections>`_, which can produce injections above a minimum SNR threshold. ::

 ##############
 # Injections #
 ##############
 
 # Change as appropriate, whitespace is important
 MCHIRP_INJECTIONS := 0.5:100.0:1_injections.xml
 # Minimum component mass 1 for injections
 INJ_MIN_MASS1 = 1.35
 # Maximum component mass 1 for injections
 INJ_MAX_MASS1 = 1.45
 # Minimum component mass 2 for injections
 INJ_MIN_MASS2 = 1.35
 # Maximum component mass 2 for injections
 INJ_MAX_MASS2 = 1.45
 # Mean component mass 1 for injections
 INJ_MEAN_MASS1 = 1.4
 # Mean component mass 2 for injections
 INJ_MEAN_MASS2 = 1.4
 # Standard dev component mass 1 for injections
 INJ_STD_MASS1 = 0.01
 # Standard dev component mass 2 for injections
 INJ_STD_MASS2 = 0.01
 # Minimum total mass for injections
 INJ_MIN_TOTAL_MASS = 2.7
 # Maximum total mass for injections
 INJ_MAX_TOTAL_MASS = 2.9
 # minimum frequency for injections. NOTE this should be lower than the intended filtering frequency
 INJ_FLOW = 15
 # Minimum injection distance in kpc
 INJ_MIN_DIST = 20000
 # Maximum injection distance in kpc
 INJ_MAX_DIST = 200000

Injection set parameters. The injection file is then produced with this command::

 lalapps_inspinj \
         --m-distr gaussian \
         --min-mass1 $(INJ_MIN_MASS1) \
         --max-mass1 $(INJ_MAX_MASS1) \
         --min-mass2 $(INJ_MIN_MASS2) \
         --max-mass2 $(INJ_MAX_MASS2) \
         --min-mtotal $(INJ_MIN_TOTAL_MASS) \
         --max-mtotal $(INJ_MAX_TOTAL_MASS) \
         --mean-mass1 $(INJ_MEAN_MASS1) \
         --mean-mass2 $(INJ_MEAN_MASS2) \
         --stdev-mass1 $(INJ_STD_MASS1) \
         --stdev-mass2 $(INJ_STD_MASS2) \
         --gps-start-time $(START) \
         --gps-end-time $(STOP) \
         --disable-spin \
         --d-distr uniform \
         --i-distr uniform \
         --min-distance $(INJ_MIN_DIST) \
         --max-distance $(INJ_MAX_DIST) \
         --waveform TaylorT4threePointFivePN \
         --l-distr random \
         --f-lower $(INJ_FLOW) \
         --time-step 20 \
         --t-distr uniform \
         --time-interval 3 \
         --seed 51056 \
         --output 1_injections.xml
 ligolw_no_ilwdchar 1_injections.xml

Generate/copy template bank and then split this into sub-banks
--------------------------------------------------------------

The next step is to acquire a template bank that will be used to filter the data. The BNS Makefile produces its own BNS template bank containing ~13,500 templates (parameters are shown below) but there are also existing template bank that can be used. If you are using a pre-existing template bank, then much of the next two code blocks can be ignored/removed, though some parameters are still used elsewhere.

**Note. lalapps_tmpltbank is deprecated code and should not be used for actual analyses.** It is used here as it is faster to run than more modern codes such as `lalapps_cbc_sbank <https://lscsoft.docs.ligo.org/lalsuite/lalapps/namespacelalapps__cbc__sbank.html>`_. ::

 ############################
 # Template bank parameters #
 ############################
 
 # Note that these can can change if you modify the template bank program.
 # Waveform approximant
 APPROXIMANT = TaylorF2
 # Minimum component mass for the template bank
 MIN_MASS = 0.99
 # Maximum component mass for the template bank
 MAX_MASS = 3.1
 # Minimum total mass for the template bank
 MIN_TOTAL_MASS = 1.98
 # Maximum total mass for the template bank
 MAX_TOTAL_MASS = 6.2
 # Maximum symmetric mass ratio for the template bank
 MAX_ETA = 0.25
 # Minimum symmetric mass ratio for the template bank
 MIN_ETA = 0.18
 # Low frequency cut off for the template bank placement
 LOW_FREQUENCY_CUTOFF = 15.0
 # High pass frequency to condition the data before measuring the psd for template placement
 HIGH_PASS_FREQ = 10.0
 # Highest frequency at which to compute the metric
 HIGH_FREQUENCY_CUTOFF = 1024.0
 # The sample rate at which to compute the template bank
 SAMPLE_RATE = 4096
 # The minimal match of the template bank; determines how much SNR is retained for signals "in between the bank points"
 MM = 0.975
 # The start time for reading the data for the bank
 BANKSTART = 1187000000
 # The stop time for reading the data for the bank (Bank start + 2048s)
 BANKSTOP = 1187002048

Template bank parameters. The bank is then produced with this command::

 lalapps_tmpltbank \
         --disable-compute-moments \
         --grid-spacing Hexagonal \
         --dynamic-range-exponent 69.0 \
         --enable-high-pass $(HIGH_PASS_FREQ) \
         --high-pass-order 8 \
         --strain-high-pass-order 8 \
         --minimum-mass $(MIN_MASS) \
         --maximum-mass $(MAX_MASS) \
         --min-total-mass $(MIN_TOTAL_MASS) \
         --max-total-mass $(MAX_TOTAL_MASS) \
         --max-eta $(MAX_ETA) \
         --min-eta $(MIN_ETA) \
         --gps-start-time $(BANKSTART) \
         --gps-end-time $(BANKSTOP) \
         --calibrated-data real_8 \
         --channel-name H1:$(H1_CHANNEL) \
         --space Tau0Tau3 \
         --number-of-segments 15 \
         --minimal-match $(MM) \
         --high-pass-attenuation 0.1 \
         --min-high-freq-cutoff ERD \
         --segment-length 1048576 \
         --low-frequency-cutoff $(LOW_FREQUENCY_CUTOFF) \
         --pad-data 8 \
         --num-freq-cutoffs 1 \
         --sample-rate $(SAMPLE_RATE) \
         --high-frequency-cutoff $(HIGH_FREQUENCY_CUTOFF) \
         --resample-filter ldas \
         --strain-high-pass-atten 0.1 \
         --strain-high-pass-freq $(HIGH_PASS_FREQ) \
         --frame-cache H1_frame.cache \
         --max-high-freq-cutoff ERD \
         --approximant $(APPROXIMANT) \
         --order twoPN \
         --spectrum-type median \
         --verbose
 ligolw_no_ilwdchar H1-TMPLTBANK-$(START)-2048.xml
 gstlal_inspiral_add_template_ids H1-TMPLTBANK-$(START)-2048.xml

After obtaining a bank ``gstlal_inspiral_add_template_ids`` needs to be run on it in order to work with the mass model used in the main analysis. ::

 mkdir -p $*_split_bank
 gstlal_bank_splitter \
         --f-low $(LOW_FREQUENCY_CUTOFF) \
         --group-by-chi $(NUM_CHI_BINS) \
         --output-path $*_split_bank \
         --approximant $(APPROXIMANT1) \
         --approximant $(APPROXIMANT2) \
         --output-cache %_split_bank.cache \
         --overlap $(OVERLAP) \
         --instrument $* \
         --n $(NUM_SPLIT_TEMPLATES) \
         --sort-by mchirp \
         --max-f-final $(HIGH_FREQUENCY_CUTOFF) \
         --write-svd-caches \
         --num-banks $(NUMBANKS) \
         H1-TMPLTBANK-$(START)-2048.xml

This program needs to be run on the template bank being used to split it up into sub banks that will be passed to the singular value decomposition code within the pipeline.

Run gstlal_inspiral_pipe to produce offline analysis dag
--------------------------------------------------------

The final stage of the Makefile that produces the analysis dag. ::

 gstlal_inspiral_pipe \
         --data-source frames \
         --gps-start-time $(START) \
         --gps-end-time $(STOP) \
         --frame-cache frame.cache \
         --frame-segments-file segments.xml.gz \
         --vetoes vetoes.xml.gz \
         --frame-segments-name datasegments  \
         --control-peak-time $(PEAK) \
         --template-bank H1-TMPLTBANK-$(START)-2048.xml \
         --num-banks $(NUMBANKS) \
         --fir-stride 1 \
         --web-dir $(WEBDIR) \
         --time-slide-file tisi.xml \
         --inj-time-slide-file inj_tisi.xml \
         $(INJECTION_LIST) \
         --bank-cache $(BANK_CACHE_STRING) \
         --tolerance 0.9999 \
         --overlap $(OVERLAP) \
         --flow $(LOW_FREQUENCY_CUTOFF) \
         $(CHANNEL_NAMES) \
         --autocorrelation-length $(AC_LENGTH) \
         $(ADDITIONAL_DAG_OPTIONS) \
         $(CONDOR_COMMANDS) \
         --ht-gate-threshold-linear 0.8:15.0-45.0:100.0 \
         --request-cpu 2 \
         --request-memory 5GB \
         --min-instruments $(MIN_IFOS) \
         --ranking-stat-samples 4194304 \
         --mass-model=ligo
 sed -i '1s/^/JOBSTATE_LOG logs\/trigger_pipe.jobstate.log\n/' trigger_pipe.dag

Additional commands and submitting the dag
------------------------------------------

There are some additional commands that can be run at the end of the Makefile to perform various tasks.  ::

 sed -i 's/.*queue.*/Requirements = regexp("Intel.*v[3-5]", TARGET.cpuinfo_model_name)\n&/' *.sub

A ``sed`` command that makes jobs only run on intel architecture. Only needed if using an optimised build. ::

 sed -i 's/.*request_memory.*/#&\n+MemoryUsage = ( 2048 ) * 2 \/ 3\nrequest_memory = ( MemoryUsage ) * 3 \/ 2\nperiodic_hold = ( MemoryUsage >= ( ( RequestMemory ) * 3 \/ 2 ) )\nperiodic_release = (JobStatus == 5) \&\& ((CurrentTime - EnteredCurrentStatus) > 180) \&\& (HoldReasonCode != 34)/' *.sub
 sed -i 's@+MemoryUsage = ( 2048 ) \* 2 / 3@+MemoryUsage = ( 6000 ) \* 2 / 3@' gstlal_inspiral.sub
 sed -i 's@+MemoryUsage = ( 2048 ) \* 2 / 3@+MemoryUsage = ( 6000 ) \* 2 / 3@' gstlal_inspiral_inj.sub

A set of ``sed`` commands to to make the memory request of jobs dynamical. These commands shouldn't be needed for most standard cases, but if you notice that jobs are being placed on hold by condor for going over their requested memory allocation, then these should allow the jobs to run. ::

 sed -i "/^environment/s?\$$?GSTLAL_FIR_WHITEN=0;?" *.sub

A ``sed`` command to set ``GSTLAL_FIR_WHITEN=0`` for all jobs. Required in all cases. This environment variable is sometimes also set within the env.sh file when sourcing an environment, if it was built by the user. This sed command should be included if using the system build. ::

 sed -i 's@environment = GST_REGISTRY_UPDATE=no;@environment = "GST_REGISTRY_UPDATE=no LD_PRELOAD=$(MKLROOT)/lib/intel64/libmkl_core.so"@g' gstlal_inspiral_injection_snr.sub

A ``sed`` command to force the use of MKL libraries for injection SNRs. Only needed if using an optimised build.

Running the Makefile
--------------------

Assuming you have all the prerequisites, running the BNS Makefile as it is only requires a few changes. These are:

 * Line 3: set accounting tag
 * Line 66: Set analysis run tag. Use this to identify different runs, e.g. TAG = BNS_test_dag_190401
 * Line 129: Set path to veto definer file
 * Line 183: Set path to Makefile.offline_analysis_rules

Then ensuring you have the correct environment set, run with: make -f Makefile.BNS_HL_test_dag_O2

Submitting the dag
------------------

Commands for submitting the dag to condor and then to monitor the status of the dag are output at the end of its running. The ``grep`` command provides some formatting to the output, removing superfluous information::

 Submit with: condor_submit_dag trigger_pipe.dag
 Monitor with: tail -f trigger_pipe.dag.dagman.out | grep -v -e ULOG -e monitoring

