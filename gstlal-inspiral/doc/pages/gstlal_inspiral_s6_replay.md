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
online environment.  Details can be found <a href="https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/140812103550GeneralData%20broadcasting">here</a>

\subsection Data Data

Some quick facts:

 - GPS Start: 968543943
 - GPS End: 971622087
 - IFOs: H1, L1

\subsection Resources Resources

 - Online: 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002
 - Offline: NEMO Cluster (does not need to be as specific as online)

\section Analysis Analysis

 - location, UWM: /home/gstlalcbc/review/s6replay
 - online and offline are in the appropriately named directories

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
 - Makefile to make the template banks

		FSTART = 871147316
		FSTOP =  871149864
		START = 871147516
		STOP =  871149564

		# Template bank parameters
		MIN_MASS = 1.0
		MAX_MASS = 12.0
		MIN_TOTAL_MASS = 2.0 
		MAX_TOTAL_MASS = 15.0
		LOW_FREQUENCY_CUTOFF = 40.0
		HIGH_PASS_FREQ = 35
		HIGH_FREQUENCY_CUTOFF = 1024.0
		SAMPLE_RATE = 2048
		NUM_SPLIT_TEMPLATES = 100
		OVERLAP = 20
		MM = 0.97
		NUMBANKS = 5,6,7

		all : bank.dag


		FAKE :
			gstlal_fake_frames --gps-start-time $(FSTART) --gps-end-time $(FSTOP) --channel-name H1=FAKE-STRAIN --verbose --data-source LIGO --frame-type FAKE --output-path FAKE

		frame.cache : FAKE
			ls FAKE/H-FAKE-871/*.gwf | lalapps_path2cache > frame.cache

		H1-TMPLTBANK-871147516-2048.xml : frame.cache
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
				--gps-start-time $(START) \
				--gps-end-time $(STOP) \
				--calibrated-data real_8 \
				--channel-name H1:FAKE-STRAIN \
				--space Tau0Tau3 \
				--number-of-segments 15 \
				--minimal-match $(MM) \
				--candle-snr 8 \
				--high-pass-attenuation 0.1 \
				--min-high-freq-cutoff SchwarzISCO \
				--segment-length 524288 \
				--low-frequency-cutoff $(LOW_FREQUENCY_CUTOFF) \
				--pad-data 8 \
				--num-freq-cutoffs 1 \
				--sample-rate $(SAMPLE_RATE) \
				--high-frequency-cutoff $(HIGH_FREQUENCY_CUTOFF) \
				--resample-filter ldas \
				--strain-high-pass-atten 0.1 \
				--strain-high-pass-freq $(HIGH_PASS_FREQ) \
				--frame-cache frame.cache \
				--max-high-freq-cutoff SchwarzISCO \
				--approximant TaylorF2 \
				--order twoPN \
				--spectrum-type median \
				--verbose

		%_split_bank.cache: H1-TMPLTBANK-871147516-2048.xml
			mkdir -p $*_split_bank
			gstlal_bank_splitter --overlap $(OVERLAP) --instrument $* --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871147516-2048.xml --output-cache $@ --output-path $*_split_bank --approximant TaylorF2

		%_bank.dag : %_split_bank.cache reference_psd.xml.gz
			cp $< tmp
			gstlal_inspiral_svd_bank_pipe --autocorrelation-length 351 --instrument $* --reference-psd reference_psd.xml.gz --bank-cache $< --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name $@ --num-banks $(NUMBANKS)

		bank.dag : H1_bank.dag L1_bank.dag
			cat H1_bank.dag L1_bank.dag > bank.dag
			rm -f H1_bank.dag L1_bank.dag

		clean :
			rm -rf *.sub* *.dag* *.cache *.sh *.xml *.gz logs gstlal_svd_bank* *split_bank

\subsubsection Triggers Triggers

 - /home/gstlalcbc/review/s6replay/online/trigs
 - Makefile to make the analysis dag

		H1_BANK_CACHE = ../bank/H1_bank.cache
		L1_BANK_CACHE = ../bank/L1_bank.cache

		H1CHANNEL=LDAS-STRAIN
		L1CHANNEL=LDAS-STRAIN
		H1INJCHANNEL=LDAS-STRAIN_CBC_INJ
		L1INJCHANNEL=LDAS-STRAIN_CBC_INJ

		H1DQCHANNEL=LSC-DATA_QUALITY_VECTOR
		L1DQCHANNEL=LSC-DATA_QUALITY_VECTOR
		H1INJDQCHANNEL=LSC-DATA_QUALITY_VECTOR
		L1INJDQCHANNEL=LSC-DATA_QUALITY_VECTOR

		H1FRAMEXMIT=224.3.2.221:7296
		L1FRAMEXMIT=224.3.2.222:7297
		H1INJFRAMEXMIT=224.3.2.224:7299
		L1INJFRAMEXMIT=224.3.2.225:7300

		dag : marginalized_likelihood.xml.gz prior.cache
			gstlal_ll_inspiral_pipe \
				--bank-cache H1=$(H1_BANK_CACHE),L1=$(L1_BANK_CACHE) \
				--likelihood-cache prior.cache \
				--channel-name=H1=$(H1CHANNEL) \
				--channel-name=L1=$(L1CHANNEL) \
				--inj-channel-name=H1=$(H1INJCHANNEL) \
				--inj-channel-name=L1=$(L1INJCHANNEL) \
				--dq-channel-name=L1=$(L1DQCHANNEL) \
				--dq-channel-name=H1=$(H1DQCHANNEL) \
				--inj-dq-channel-name=L1=$(L1INJDQCHANNEL) \
				--inj-dq-channel-name=H1=$(H1INJDQCHANNEL) \
				--framexmit-addr=H1=$(H1FRAMEXMIT) \
				--framexmit-addr=L1=$(L1FRAMEXMIT) \
				--inj-framexmit-addr=H1=$(H1INJFRAMEXMIT) \
				--inj-framexmit-addr=L1=$(L1INJFRAMEXMIT) \
				--framexmit-iface=172.16.10.1 \
				--inj-framexmit-iface=172.16.10.1 \
				--state-vector-on-bits=H1=0x1 \
				--state-vector-on-bits=L1=0x1 \
				--inj-state-vector-on-bits=H1=1 \
				--inj-state-vector-on-bits=L1=1 \
				--state-vector-off-bits=H1=2 \
				--state-vector-off-bits=L1=2 \
				--inj-state-vector-off-bits=H1=2 \
				--inj-state-vector-off-bits=L1=2 \
				--gracedb-far-threshold 0.0001 \
				--inj-gracedb-far-threshold 0.0001 \
				--control-peak-time 0 \
				--fir-stride 1 \
				--marginalized-likelihood-file marginalized_likelihood.xml.gz \
				--gracedb-group Test \
				--gracedb-search LowMass \
				--inj-gracedb-group CBC \
				--inj-gracedb-search ReplayLowMassInj \
				--thinca-interval 1 \
				--ht-gate-threshold 15 \
				--data-source framexmit \
				--likelihood-snapshot-interval 14400 \
				--lvalert-listener-program gstlal_inspiral_followups_from_gracedb \
				--lvalert-listener-program gstlal_inspiral_lvalert_psd_plotter \
				--inj-lvalert-listener-program gstlal_inspiral_followups_from_gracedb \
				--inj-lvalert-listener-program gstlal_inspiral_lvalert_psd_plotter \
				--inspiral-condor-command '+Online_CBC_SVD=True' \
				--inspiral-condor-command 'Requirements=(TARGET.Online_CBC_SVD=?=True)' \
				--inspiral-condor-command 'request_cpus=4' \
				--inspiral-condor-command 'request_memory=8000'

		set-far-thresh :
			gstlal_ll_inspiral_gracedb_threshold \
				--gracedb-far-threshold $(FINAL_FAR_THRESH) \
				*registry.txt

		prior.cache :
			gstlal_ll_inspiral_create_prior_diststats \
				--write-likelihood-cache $@ \
				--segment-and-horizon=H1:1000000000:1000000100:40 \
				--segment-and-horizon=L1:1000000000:1000000100:40 \
				--num-banks $(shell wc -l $(H1_BANK_CACHE) | awk '{print $1}') \
				--verbose

		marginalized_likelihood.xml.gz : prior.cache
			gstlal_inspiral_marginalize_likelihood \
				--output $@ \
				--verbose \
				--likelihood-cache $<

		clean :
			rm -rf gstlal_inspiral gstlal_inspiral_inj gracedb gstlal_inspiral_marginalize_likelihoods_online gstlal_ll_inspiral_get_urls lvalert_listen 
			rm -rf *.txt lvalert.ini *.gz trigger_pipe.* *.sub logs lvalert*.sh node* *.xml prior.cache

\subsection Offline Offline Analysis


\subsubsection OfflineAnalysis Offline Analysis

 - /home/gstlalcbc/review/s6replay/offline
 - Makefile which contains rules for every offline analysis

		# Misc useful definitions
		empty:=
		space:= $(empty) $(empty)
		comma:= ,

		# the point of this is to build the string e.g. H1=../bank/H1_bank.cache,L1=../bank/L1_bank.cache
		BANK_CACHE_PREFIX = $(empty)
		BANK_CACHE_SUFFIX = _split_bank.cache
		BANK_CACHE_FILES = $(addsuffix $(BANK_CACHE_SUFFIX),$(IFOS))
		BANK_CACHE_STRING:= $(addprefix $(BANK_CACHE_PREFIX),$(IFOS))
		BANK_CACHE_STRING:= $(addprefix =,$(BANK_CACHE_STRING))
		BANK_CACHE_STRING:= $(addsuffix $(BANK_CACHE_SUFFIX),$(BANK_CACHE_STRING))
		BANK_CACHE_STRING:= $(join $(IFOS),$(BANK_CACHE_STRING))
		BANK_CACHE_STRING:= $(strip $(BANK_CACHE_STRING))
		BANK_CACHE_STRING:= $(subst $(space),$(comma),$(BANK_CACHE_STRING))

		# Segments file names
		segments_suffix := _segmentspadded.xml
		SEGMENTS_FILES  := $(addsuffix $(segments_suffix),$(IFOS))

		# Frame cache file names
		frame_suffix      := _frame.cache
		FRAME_CACHE_FILES := $(addsuffix $(frame_suffix),$(IFOS))

		# Injection file names
		injections:=--injections $(space)
		INJECTION_LIST := $(subst $(space), $(injections), $(INJECTIONS))

 - Makefile to create the analysis dag using the template bank generated in the online analysis, H1-TMPLTBANK-871147516-2048.xml

		#
		# Template bank parameters
		#

		# The filtering start frequency
		LOW_FREQUENCY_CUTOFF = 40.0
		# The maximum frequency to filter to
		HIGH_FREQUENCY_CUTOFF = 1024.0
		# Controls the number of templates in each SVD sub bank
		NUM_SPLIT_TEMPLATES = 100
		# Controls the overlap from sub bank to sub bank - helps mitigate edge effects
		# in the SVD.  Redundant templates will be removed
		OVERLAP = 20
		# The approximant that you wish to filter with
		APPROXIMANT = TaylorF2

		#
		# Triggering parameters
		#

		# The detectors to analyze
		IFOS = H1 L1
		# The GPS start time of the S6 replay
		START = 967161687
		# The GPS end time of the S6 replay
		STOP = 968371287
		# A user tag for the run
		TAG = offline_s6_replay
		# A web directory for output
		WEBDIR = ~/public_html/$(TAG)
		# The number of sub banks to process in parallel for each gstlal_inspiral job
		NUMBANKS = 5,6,7
		# The control peak time for the composite detection statistic.  If set to 0 the
		# statistic is disabled
		PEAK = 0
		# The length of autocorrelation chi-squared in sample points
		AC_LENGTH = 351
		# The minimum number of samples to include in a given time slice
		SAMPLES_MIN = 1024 # default value
		# The maximum number of samples to include in the 256 Hz or above time slices
		SAMPLES_MAX_256 = 1024 # default value

		#
		# additional options, e.g.,
		#

		#ADDITIONAL_DAG_OPTIONS = "--blind-injections BNS-MDC1-WIDE.xml"

		#
		# Injections
		#

		# The seed is the string before the suffix _injections.xml
		# Change as appropriate, whitespace is important
		INJECTIONS := S6_bns_injs_shifted.xml

		#
		# Segment and frame type info
		#

		# The LIGO and Virgo frame types
		LIGO_FRAME_TYPE_SUFFIX='LDAS_C02_L2'
		# The Channel names. FIXME sadly you have to change the CHANNEL_NAMES string if
		# you want to analyze a different set of IFOS
		H1_CHANNEL=LDAS-STRAIN
		L1_CHANNEL=LDAS-STRAIN
		CHANNEL_NAMES:=--channel-name=H1=$(H1_CHANNEL) --channel-name=L1=$(L1_CHANNEL)

		#
		# Get some basic definitions.  NOTE this comes from the share directory probably.
		#

		include Makefile.offline_analysis_rules

		#
		# Workflow
		#

		all : dag

		H1-TMPLTBANK-871147516-2048.xml :
			cp ../online/bank/H1-TMPLTBANK-871147516-2048.xml .

		%_split_bank.cache : H1-TMPLTBANK-871147516-2048.xml
			mkdir -p $*_split_bank
			gstlal_bank_splitter --f-low $(LOW_FREQUENCY_CUTOFF) --group-by-chi --output-path $*_split_bank --approximant $(APPROXIMANT) --output-cache $@ --overlap $(OVERLAP) --instrument $* --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) $<

		plots :
			mkdir plots

		$(WEBDIR) : 
			mkdir -p $(WEBDIR)

		tisi.xml :
			ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=0:0:0 --instrument=V1=0:0:0 tisi0.xml
			ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=3.14159:3.14159:3.14159 --instrument=V1=7.892:7.892:7.892 tisi1.xml
			ligolw_add --output $@ tisi0.xml tisi1.xml

		dag : segments.xml.gz vetoes.xml.gz frame.cache tisi.xml plots $(WEBDIR) $(INJECTIONS) $(BANK_CACHE_FILES)
			gstlal_inspiral_pipe --data-source frames --gps-start-time $(START) --gps-end-time $(STOP) --frame-cache frame.cache --frame-segments-file segments.xml.gz --vetoes vetoes.xml.gz --frame-segments-name datasegments  --control-peak-time $(PEAK) --num-banks $(NUMBANKS) --fir-stride 4 --web-dir $(WEBDIR) --time-slide-file tisi.xml $(INJECTION_LIST) --bank-cache $(BANK_CACHE_STRING) --tolerance 0.9999 --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) $(CHANNEL_NAMES) --autocorrelation-length $(AC_LENGTH) --samples-min $(SAMPLES_MIN) --samples-max-256 $(SAMPLES_MAX_256) $(ADDITIONAL_DAG_OPTIONS)

		V1_frame.cache:
			# FIXME force the observatory column to actually be instrument
			ligo_data_find -o V -t $(VIRGO_FRAME_TYPE) -l  -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" V1_"$$2" "$$3" "$$4" "$$5}' > $@

		%_frame.cache:
			# FIXME horrible hack to get the observatory, not guaranteed to work
			$(eval OBS:=$*)
			$(eval OBS:=$(subst 1,$(empty),$(OBS)))
			$(eval OBS:=$(subst 2,$(empty),$(OBS)))
			# FIXME force the observatory column to actually be instrument
			ligo_data_find -o $(OBS) -t $*_$(LIGO_FRAME_TYPE_SUFFIX) -l  -s $(START) -e $(STOP) --url-type file | awk '{ print $$1" $*_"$$2" "$$3" "$$4" "$$5}' > $@

		frame.cache: $(FRAME_CACHE_FILES)
			cat $(FRAME_CACHE_FILES) > frame.cache

		segments.xml.gz: frame.cache
			# These segments come from the MDC set  
			gsiscp pcdev3.cgca.uwm.edu:/home/channa/public_html/SELECTED_SEGS.xml.gz $@
			gstlal_cache_to_segments frame.cache nogaps.xml
			gstlal_segments_operations --segment-file1 $@ --segment-file2 nogaps.xml --intersection --output-file $@
			-rm -vf nogaps.xml
			gstlal_segments_trim --trim 8 --gps-start-time $(START) --gps-end-time $(STOP) --min-length 2048 --output $@ $@

		vetoes.xml.gz:
			gsiscp pcdev3.cgca.uwm.edu:/home/channa/public_html/COMBINED_CAT_4_VETO_SEGS.xml.gz $@
			gstlal_segments_trim --gps-start-time $(START) --gps-end-time $(STOP) --segment-name vetoes --output $@ $@

		clean:
			-rm -rvf *.sub *.dag* *.cache *.sh logs *.sqlite plots *.html Images *.css *.js
			-rm -rvf lalapps_run_sqlite/ ligolw_* gstlal_*
			-rm -vf segments.xml.gz tisi.xml H*.xml L*.xml V*.xml ?_injections.xml ????-*_split_bank-*.xml vetoes.xml.gz
			-rm -vf *marginalized*.xml.gz *-ALL_LLOID*.xml.gz
			-rm -vf tisi0.xml tisi1.xml
			-rm -rf *_split_bank

\section Results Results
 - <a href="https://gracedb.ligo.org/events/search/?query=test%20gstlal%20lowmass%201118438426..1119648026">GraceDb query</a>
 - <a href="https://simdb.phys.uwm.edu/events/search/?query=cbc%20gstlal%20replaylowmassinj%201118438426..1119648026">SimDb query</a>
 - <a href="https://simdb.phys.uwm.edu/events/search/?query=cbc%20hardwareinjection%20replaylowmassinj%201118438426..1119648026">SimDb injection file query</a>
 - <a href="https://ldas-jobs.cgca.uwm.edu/~gstlalcbc/range.png">Low latency sensitivity plots</a>
   - Covers the last 48 hours
   - Updated every 5-10 minutes
