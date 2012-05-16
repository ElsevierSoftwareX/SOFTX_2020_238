FSTART = 871147316
FSTOP =  871149864
START = 871147516
STOP =  871149564
IFOS = H

# Template bank parameters
MIN_MASS = 1.2
MAX_MASS = 1.6
MIN_TOTAL_MASS = 2.4
MAX_TOTAL_MASS = 3.2
LOW_FREQUENCY_CUTOFF = 14.0
HIGH_PASS_FREQ = 9
HIGH_FREQUENCY_CUTOFF = 2047.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 250
OVERLAP = 50
MM = 0.99

all : dag

lalapps_path2cache :
	wget 'http://www.lsc-group.phys.uwm.edu/cgit/lalsuite/plain/pylal/bin/lalapps_path2cache'
	chmod +x lalapps_path2cache

frame.cache : lalapps_path2cache
	gstlal_fake_aligo_frames --gps-start-time $(FSTART) --gps-end-time $(FSTOP) --instrument H1 --verbose --fake-data AdvLIGO --frame-type test
	ls H1/H-H1_test-8711/*.gwf | ./lalapps_path2cache > frame.cache

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
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1 \
		--candle-mass2 1 \
		--channel-name H1:FAKE-STRAIN \
		--space Tau0Tau3 \
		--number-of-segments 15 \
		--minimal-match $(MM) \
		--candle-snr 8 \
		--debug-level 33 \
		--high-pass-attenuation 0.1 \
		--min-high-freq-cutoff SchwarzISCO \
		--segment-length 1048576 \
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
		--verbose \
		&& gstlal_bank_splitter --overlap $(OVERLAP) -i H2 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871147516-2048.xml \
		&& gstlal_bank_splitter --overlap $(OVERLAP) -i H1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871147516-2048.xml \
		&& gstlal_bank_splitter --overlap $(OVERLAP) -i V1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871147516-2048.xml \
		&& gstlal_bank_splitter --overlap $(OVERLAP) -i L1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871147516-2048.xml

H1psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output H1psd.xml.gz ZERO_DET_high_P.txt

H2psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H2 --output H2psd.xml.gz ZERO_DET_high_P.txt

L1psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument L1 --output L1psd.xml.gz ZERO_DET_high_P.txt

V1psd.xml.gz :
	cp virgo_online_psd.xml.gz V1psd.xml.gz 

reference_psd.xml: H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz
	ligolw_add --output reference_psd.xml H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz

dag : H1-TMPLTBANK-871147516-2048.xml reference_psd.xml lalapps_path2cache
	ls *H2_split_bank-H1-TMPLTBANK-871147516-2048.xml | ./lalapps_path2cache > H2_split_bank.cache
	ls *H1_split_bank-H1-TMPLTBANK-871147516-2048.xml | ./lalapps_path2cache > H1_split_bank.cache
	ls *V1_split_bank-H1-TMPLTBANK-871147516-2048.xml | ./lalapps_path2cache > V1_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871147516-2048.xml | ./lalapps_path2cache > L1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --instrument H2 --reference-psd reference_psd.xml --bank-cache H2_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H2_bank
	gstlal_inspiral_svd_bank_pipe --instrument H1 --reference-psd reference_psd.xml --bank-cache H1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H1_bank
	gstlal_inspiral_svd_bank_pipe --instrument V1 --reference-psd reference_psd.xml --bank-cache V1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name V1_bank
	gstlal_inspiral_svd_bank_pipe --instrument L1 --reference-psd reference_psd.xml --bank-cache L1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name L1_bank
	cat H2_bank.dag H1_bank.dag L1_bank.dag V1_bank.dag  > bank.dag

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871147516-2048.xml *split_bank-H1-TMPLTBANK-871147516-2048.xml H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz segments.H.xml logs
