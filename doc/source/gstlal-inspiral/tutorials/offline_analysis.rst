Documentation for running an offline compact binary coalescence analysis
========================================================================

Prerequisites
-------------

 - Fully functional gstlal, gstlal-ugly, gstlal-inspiral installation
 - Condor managed computing resource using the LIGO Data Grid configuration
 - Access to gravitational wave data, stored locally or via CVMFS

Introduction
------------

This tutorial will help you to setup and run a offline gravitational wave search for binary neutron stars. The information contained within this document can easily be modified to perform a wide range of searches.

The offline analysis has a somewhat involved setup procedure. This documentation covers all of it. The analysis itself is performed by a pipeline contained within a dag (Directed Acyclic Graph) that is managed by condor. The dag and job sub files are produced by running gstlal_inspiral_pipe. This program requires several input files that are produced in several steps, all of which are detailed below. These input files are:

 * segments.xml.gz
 * vetoes.xml.gz
 * frame.cache
 * inj_tisi.xml
 * tisi.xml
 * injection file
 * split bank cache files

The steps to produce the full analysis dag file are:

 1. Analysis variables defined at the top of offline Makefile
 2. Generate frame cache, segments, vetoes, and tisi files
 3. Generate/copy template bank and then split this into sub-banks
 4. Run gstlal_inspiral_pipe to produce offline analysis dag

The information contained within this page is based off the O2 BNS test dag, an offline analysis focused on 100,000s centered around GW170817. The dag used to perform the analysis can be produced using a `Makefile <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/O3/offline/O2/Makefile.BNS_HL_test_dag_O2>`_ that generats most of the required files. This tutorial will just cover the HL detector pair configureation, though a HLV Makefile can be found `here <https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/O3/offline/O2/Makefile.BNS_HLV_test_dag_O2>`_. In this tutorial we detail each stage of the Makefile needed to run an offline analysis.

Analysis variables defined at the top of offline Makefile
---------------------------------------------------------

There many variables that are set at the top of the offline Makefile. Some of these should not be changed unless you know what you are doing. The variables that should be changed/set are explained here::

 ACCOUNTING_TAG=ligo.dev.o3.cbc.uber.gstlaloffline

An accounting tag used to measure LDG computational use. See https://ldas-gridmon.ligo.caltech.edu/ldg_accounting/user. ::

 GROUP_USER=albert.einstein

This should be your alber.einstein user idenification. This is only needed if using a shared account. ::

 IFOS = H1 L1
 MIN_IFOS = 2

Define which detectors to include within the analysis. H1, L1, and V1 are currently supported. Set minimum number of operational dectors for which to analyise. Able to analyse single detector time. ::

 START = 1187000000
 STOP = 1187100000

Set start and stop time of the analysis in GPS seconds. The times stated here are 100,000s around GW170817. See https://www.gw-openscience.org/gps/ for conversions. ::

 TAG = BNS_test_dag
 RUN = run_1
 WEBDIR = ~/public_html/testing/$(TAG)/$(START)-$(STOP)-$(RUN)

Set output directory for summary page of results. ::

 MCHIRP_INJECTIONS := 0.5:100.0:1_injections.xml

Used to specify injection file, and chirpmass range over which to filter it. Multiple injection files can be given at once, these should be space seperated, with no whitespace at the end of the line. ::

 VETODEF = /path/to/H1L1-CBC_VETO_DEFINER_CLEANED_C02_O2_1164556817-23176801.xml

Veto definer file. Used to determine what data to veto. See https://git.ligo.org/detchar/veto-definitions/tree/master/cbc for all veto definer files. ::

 # GSTLAL_SEGMENTS Options
 SEG_SERVER=https://segments.ligo.org
 # C02 cleaned
 LIGO_SEGMENTS="$*:DCH-CLEAN_SCIENCE_C02:1"

 # The LIGO frame types
 # C02 cleaened
 HANFORD_FRAME_TYPE='H1_CLEANED_HOFT_C02'
 LIVINGSTON_FRAME_TYPE='L1_CLEANED_HOFT_C02'

 # The Channel names.
 # C02 cleaned
 H1_CHANNEL=DCH-CLEAN_STRAIN_C02
 L1_CHANNEL=DCH-CLEAN_STRAIN_C02

Gravitational wave data segment, frame type, and channel name information. See https://wiki.ligo.org/LSC/JRPComm/ for full information about all observing runs. ::

 include /path/to/Makefile.offline_analysis_rules

Full path to [Makefile.offline_analysis_rules](https://git.ligo.org/lscsoft/gstlal/blob/master/gstlal-inspiral/share/Makefile.offline_analysis_rules). This file contains sets of ruls for string parsing/manipulation used within the main Makefile and an up-to-date version must be included.


Generate segments, vetoes, frame cache, and tisi files
------------------------------------------------------

Generating frame.cache file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frame.cache file contains the full paths to the Gravitational Wave data .gwf files using the following format:  
Detector site identfier, frame type, start GPS time, duration, full path to file ::

 H H1__H1_CLEANED_HOFT_C02 1186998263 4096 file://localhost/hdfs/frames/O2/hoft_C02_clean/H1/H-H1_CLEANED_HOFT_C02-11869/H-H1_CLEANED_HOFT_C02-1186998263-4096.gwf


If the .gwf data files are stored locally, then you can produce individuel detector frame cache files with::

 gw_data_find -o H -t $(HANFORD_FRAME_TYPE) -l -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" $*_"$$2" "$$3" "$$4" "$$5}' > H1_frame.cache
 gw_data_find -o L -t $(LIVINGSTON_FRAME_TYPE) -l -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" $*_"$$2" "$$3" "$$4" "$$5}' > L1_frame.cache

The awk command provides some formating to put the output in the required format.

If the data must be accessed via CVMFS then the following option needs to be added to the gw_data_find arguments::

 --server datafind.ligo.org:443

And then create a combined frame.cache file with some additional formating::

 cat H1_frame.cache L1_frame.cache > frame.cache
 sed -i s/H\ $(LIGO_FRAME_TYPE)/H\ H1_$(LIGO_FRAME_TYPE)/g frame.cache
 sed -i s/L\ $(LIGO_FRAME_TYPE)/L\ L1_$(LIGO_FRAME_TYPE)/g frame.cache

Generating segments.xml.gz and vetoes.xml.gz files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The segments.xml.gz file contains a list of all data segments that should be analyised. The vetoes.xml.gz file contains a list of all data segments that should be ignored. ::

 ligolw_segment_query_dqsegdb --segment-url=${SEG_SERVER} -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$(LIGO_SEGMENTS) --result-name=datasegments > %_segmentspadded.xml
 ligolw_no_ilwdchar $*_segmentspadded.xml

This returns an initial segments list. This command makes use of some Makefile variables segmentspadded files for each detector specified by $IFOS. ligolw_no_ilwdchar is run on the output files to convert some table column types from ilwd:char to int4s. This command will beed to be run on any xml file produced by a non-gstlal program. ::

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

Combine initial segment files with CAT1 vetoe times removed. ::

 ./lauras_txt_files_to_xml -i $* -c -o $*-gates.xml $*-GATES-1163203217-24537601.txt
 ligolw_no_ilwdchar $*-gates.xml
 gstlal_segments_operations --union --segment-name VETO_CAT3_CUMULATIVE --output-file %_vetoes.xml.tmp --output-segment-name vetoes $*-VETOTIME_CAT3-*.xml $*-VETOTIME_CAT3-*.xml
 gstlal_segments_operations --union --segment-name vetoes --output-file %_vetoes.xml --output-segment-name vetoes %_vetoes.xml.tmp $*-gates.xml

Include gating times into CAT3 veto times files. ::

 ligolw_add --output vetoes.xml.gz $(VETOES_FILES)
 ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time vetoes.xml.gz
 gzip vetoes.xml.gz

Combine all vetoe files into single vetoes.xml.gz file.

Generating tisi.xml.gz and inj_tisi.xml.gz file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

 lalapps_gen_timeslides --instrument=H1=0:0:0 --instrument=L1=0:0:0 inj_tisi.xml

Generate injection time slides file. ::

 lalapps_gen_timeslides --instrument=H1=0:0:0 --instrument=L1=25.13274:25.13274:25.13274 bg_tisi.xml
 ligolw_add --output tisi.xml bg_tisi.xml inj_tisi.xml

Generate analysis time slides file.


Generate/copy template bank and then split this into sub-banks
--------------------------------------------------------------

The next step is to aquire a template bank that will be used to filter the data. The BNS Makefile produces its own BNS template bank containing ~13,500 templates (parametters are shown below) but there are also existing template bank that can be used. If you are using a pre-existing template bank, then much of the next two sections can be ignored/removed. ::

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

lalapps_tmpltbank is a rather old program and newer ones exist, such as lalapps_cbc_sbank. Which ever program you use to generate the bank, gstlal_inspiral_add_template_ids needs to be run on it in order to work with the mass model used in the main analysis. ::

 mkdir -p $*_split_bank
 gstlal_bank_splitter \
        --f-low $(LOW_FREQUENCY_CUTOFF) \
        --group-by-chi $(NUM_CHI_BINS) \
        --output-path $*_split_bank \
        --approximant $(APPROXIMANT1) \
        --approximant $(APPROXIMANT2) \
        --output-cache $@ \
        --overlap $(OVERLAP) \
        --instrument $* \
        --n $(NUM_SPLIT_TEMPLATES) \
        --sort-by mchirp \
        --max-f-final $(HIGH_FREQUENCY_CUTOFF) \
        --write-svd-caches \
        --num-banks $(NUMBANKS) \
        H1-TMPLTBANK-$(START)-2048.xml

This program needs to be run on the template bank being used to split it up into sub banks that will be passed to the singular value decompositon code within the pipeline.

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

There are some additional commands and output that are/can be run at the end of the Makefile to perform various tasks. ::

 sed -i 's/.*queue.*/Requirements = regexp("Intel.*v[3-5]", TARGET.cpuinfo_model_name)\n&/' *.sub

A sed command that makes jobs only run on intel arcatechture. Only needed if using an optimised build. ::

 sed -i 's/.*request_memory.*/#&\n+MemoryUsage = ( 2048 ) * 2 \/ 3\nrequest_memory = ( MemoryUsage ) * 3 \/ 2\nperiodic_hold = ( MemoryUsage >= ( ( RequestMemory ) * 3 \/ 2 ) )\nperiodic_release = (JobStatus == 5) \&\& ((CurrentTime - EnteredCurrentStatus) > 180) \&\& (HoldReasonCode != 34)/' *.sub
 sed -i 's@+MemoryUsage = ( 2048 ) \* 2 / 3@+MemoryUsage = ( 6000 ) \* 2 / 3@' gstlal_inspiral.sub
 sed -i 's@+MemoryUsage = ( 2048 ) \* 2 / 3@+MemoryUsage = ( 6000 ) \* 2 / 3@' gstlal_inspiral_inj.sub

A set of sed commands to to make the memory requet of jobs dynamical. These commands shouldn't be needed for most standard cases, but if you notice that jobs are being placed on hold by condor for going over their requested memory allowcation, then these should allow the jobs to run. ::

 sed -i "/^environment/s?\$$?GSTLAL_FIR_WHITEN=0;?" *.sub

A sed command to set 'GSTLAL_FIR_WHITEN=0' for all jobs. Required in all cases. This environment variable is sometimes also set within the env.sh file when sourcing an enviroment, if it was built by the user. This sed command should be included if using the system build. ::

 sed -i 's@environment = GST_REGISTRY_UPDATE=no;@environment = "GST_REGISTRY_UPDATE=no LD_PRELOAD=$(MKLROOT)/lib/intel64/libmkl_core.so"@g' gstlal_inspiral_injection_snr.sub

A sed command to force the use of MKL libraries for injection SNRs. Only needed if using an optimised build. ::

 Submit with: condor_submit_dag trigger_pipe.dag
 Monitor with: tail -f trigger_pipe.dag.dagman.out | grep -v -e ULOG -e monitoring

Commands for submitting the dag to condor and then to monitor the status of the dag. The grep command provides some formatting to the output, removing superfluous information.

Running the Makefile
--------------------

Assuming you have all the prerequisites, running the BNS Makefile as it is only requires a few changes. These are:

 * Line 3: set accounting tag
 * Line 66: Set analysis run tag. Use this to identify different runs, e.g. TAG = BNS_test_dag_190401
 * Line 129: Set path to veto definer file
 * Line 183: Set path to Makefile.offline_analysis_rules

Then to run it, ensuring you have the correct envirnment set, run with: make -f Makefile.BNS_HL_test_dag_O2

