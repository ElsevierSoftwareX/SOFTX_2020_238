START = 871147452
STOP =  871247452
IFOS = H

# Template bank parameters
MIN_MASS = 0.2
MAX_MASS = 1.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 1023.0
SAMPLE_RATE = 2048
NUM_SPLIT_TEMPLATES = 500
OVERLAP = 50
MM = 0.95

all : dag

frame.H.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H.cache
	./cache_to_segments frame.H.cache segments.H.xml

H1-TMPLTBANK-871154847-2048.xml: frame.H.cache 
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
		--gps-start-time 871154847 \
		--gps-end-time 871156895 \
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1 \
		--candle-mass2 1 \
		--channel-name H1:LSC-STRAIN \
		--space Tau0Tau3 \
		--number-of-segments 15 \
		--minimal-match $(MM) \
		--candle-snr 8 \
		--debug-level 33 \
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
		--frame-cache $^ \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--order twoPN \
		--verbose \
		--spectrum-type LIGO \
		&& ./bank_splitter --overlap $(OVERLAP) -i H1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml \
		&& ./bank_splitter --overlap $(OVERLAP) -i H2 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml \
		&& ./bank_splitter --overlap $(OVERLAP) -i L1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml

mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz : 
	./psd_xml_from_asd_txt H1 mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.txt

mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz : 
	./psd_xml_from_asd_txt H2 mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.txt

mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz : 
	./psd_xml_from_asd_txt L1 mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.txt

reference_psd.xml: mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz
	ligolw_add mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz | gzip > reference_psd.xml.gz

sql :
	cat uniqueify_coinc_definer.sql uniqueify_time_slide.sql cluster.sql > simplify_and_cluster.sql
	cat uniqueify_sim_inspiral.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql 

dag : H1-TMPLTBANK-871154847-2048.xml reference_psd.xml sql
	ls *H1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *H2_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H2_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > L1_split_bank.cache
	./svd_bank_pipe H1 mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz H1_split_bank.cache $(OVERLAP)
	./svd_bank_pipe H2 mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz H2_split_bank.cache $(OVERLAP)
	./svd_bank_pipe L1 mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz L1_split_bank.cache $(OVERLAP)
	cat mode_H1-TMPLTBANK-869040003-2048_svd_bank.dag  mode_H2-TMPLTBANK-875211248-2048_svd_bank.dag  mode_L1-TMPLTBANK-858087370-2048_svd_bank.dag > bank.dag

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml *split_bank-H1-TMPLTBANK-871154847-2048.xml mode_*.xml.gz segments.H.xml logs
