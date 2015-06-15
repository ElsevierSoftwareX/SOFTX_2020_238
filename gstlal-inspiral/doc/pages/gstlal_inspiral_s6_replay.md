\page gstlalinspirals6replaypage S6 Replay Documentation

[TOC]

\section Introduction Introduction

With O1 quickly approaching in mid to late 2015, the CBC group is doing a live
data simulation run using S6 data to test various piplines for the purpose of
code review.  This documentation page is specific to the gstlal portion of the
simulation run.  

\subsection Goals

 - Establish the accuracy of FAP in online analysis for low mass systems
 - Establish the online analysis has the appropriate sensitivity

\section Proposal Proposed Approach

The gstlal analysis team proposes to use two weeks of S6 data replayed in an
online environment.  Details can be found <a href="https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/S6VSR3ReplayMDC/140812103550GeneralData%20broadcasting">here</a>

\subsection Data

Some quick facts:

 - GPS Start: 968543943
 - GPS End: 971622087

\subsection Resources

 - 96 HT cores (48 physical cores) on three nodes: execute1000, execute1001, execute1002,

\section Analysis

 - location, UWM: /home/gstlalcbc/review/s6replay
 - online and offline are in the appropriately named directories

\subsection Analysis codes

 - gstlal 1a44f7af0cf69293f4b0883e4e4142ce263e86f4
 - all other dependencies from ER7 releases

\subesection Template Banks

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

\subsection Triggers

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

