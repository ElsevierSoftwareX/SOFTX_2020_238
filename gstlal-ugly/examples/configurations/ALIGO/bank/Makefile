CACHE = ../frames/frame.H1.cache
IFOS = H

# Template bank parameters
MIN_MASS = 1.2
MAX_MASS = 1.6
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 15.0
HIGH_PASS_FREQ = 14.0
HIGH_FREQUENCY_CUTOFF = 2047.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 300
OVERLAP = 50
MM = 0.97

all : dag

H1-TMPLTBANK-871147516-2048.xml:
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
		--gps-start-time 871147516 \
		--gps-end-time 871149564 \
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
		--segment-length 1048576 \
		--low-frequency-cutoff $(LOW_FREQUENCY_CUTOFF) \
		--pad-data 8 \
		--num-freq-cutoffs 1 \
		--sample-rate $(SAMPLE_RATE) \
		--high-frequency-cutoff $(HIGH_FREQUENCY_CUTOFF) \
		--resample-filter ldas \
		--strain-high-pass-atten 0.1 \
		--strain-high-pass-freq $(HIGH_PASS_FREQ) \
		--frame-cache $(CACHE) \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--order twoPN \
		--spectrum-type median \
		--verbose \
		&& gstlal_add_spins_to_bank --input H1-TMPLTBANK-871147516-2048.xml --output tmp.xml --chi -0.30 --chi -0.27 --chi -0.24 --chi -0.21 --chi -0.18 --chi -0.15 --chi -0.12 --chi -0.09 --chi -0.06 --chi -0.03 --chi 0.00 --chi 0.03 --chi 0.06 --chi 0.09 --chi 0.12 --chi 0.15 --chi 0.18 --chi 0.21 --chi 0.24 --chi 0.27 --chi 0.30 \
		&& mv tmp.xml H1-TMPLTBANK-871147516-2048.xml \
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
	gstlal_psd_xml_from_asd_txt --instrument V1 --output V1psd.xml.gz ZERO_DET_high_P.txt

reference_psd.xml: H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz
	ligolw_add --ilwdchar-compat --output reference_psd.xml H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz

sql :
	cat fix_coinc_instruments.sql uniqueify_coinc_definer.sql uniqueify_time_slide.sql > simplify.sql
	cat simplify.sql cluster.sql > simplify_and_cluster.sql
	cat simplify.sql delete_by_far.sql > simplify_and_delete_by_far.sql
	cat fix_coinc_instruments.sql drop_sim.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql

dag : H1-TMPLTBANK-871147516-2048.xml reference_psd.xml sql
	ls *H1_split_bank-H1-TMPLTBANK-871147516-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *V1_split_bank-H1-TMPLTBANK-871147516-2048.xml | lalapps_path2cache > V1_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871147516-2048.xml | lalapps_path2cache > L1_split_bank.cache
	./svd_bank_pipe H1 reference_psd.xml H1_split_bank.cache $(OVERLAP) $(LOW_FREQUENCY_CUTOFF)
	./svd_bank_pipe V1 reference_psd.xml V1_split_bank.cache $(OVERLAP) $(LOW_FREQUENCY_CUTOFF)
	./svd_bank_pipe L1 reference_psd.xml L1_split_bank.cache $(OVERLAP) $(LOW_FREQUENCY_CUTOFF)
	cat H1_reference_psd_svd_bank.dag L1_reference_psd_svd_bank.dag V1_reference_psd_svd_bank.dag > bank.dag

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871147516-2048.xml *split_bank-H1-TMPLTBANK-871147516-2048.xml *psd.xml.gz segments.H.xml logs
