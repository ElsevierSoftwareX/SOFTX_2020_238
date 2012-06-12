START = 871147452
STOP =  871247452
IFOS = H

# Template bank parameters
MIN_MASS = 1.1
MAX_MASS = 1.7
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 2048.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 500
MM = 0.97

EPSILON = 0.02

H1BASENAME = mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4
H2BASENAME = mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4
L1BASENAME = mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4

all : dag

frame.H.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H.cache


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
		--segment-length 1048576 \
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
		&& gstlal_bank_splitter -i H1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) $@ \
		&& gstlal_bank_splitter -i H2 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) $@ \
		&& gstlal_bank_splitter -i L1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) $@ \

${H1BASENAME}.xml.gz :
	gstlal_psd_xml_from_asd_txt --instrument H1 --output ${H1BASENAME}.xml.gz ${H1BASENAME}.txt

${H2BASENAME}.xml.gz :
	gstlal_psd_xml_from_asd_txt --instrument H2 --output ${H2BASENAME}.xml.gz ${H2BASENAME}.txt

${L1BASENAME}.xml.gz :
	gstlal_psd_xml_from_asd_txt --instrument L1 --output ${L1BASENAME}.xml.gz ${L1BASENAME}.txt

reference_psd.xml.gz: ${H1BASENAME}.xml.gz ${H2BASENAME}.xml.gz ${L1BASENAME}.xml.gz
	ligolw_add --output reference_psd.xml.gz ${H1BASENAME}.xml.gz ${H2BASENAME}.xml.gz ${L1BASENAME}.xml.gz


sql :
	cat uniqueify_coinc_definer.sql uniqueify_time_slide.sql > simplify.sql
	cat simplify.sql cluster.sql > simplify_and_cluster.sql
	cat simplify.sql delete_by_far.sql > simplify_and_delete_by_far.sql
	cat drop_sim.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql

dag : H1-TMPLTBANK-871154847-2048.xml reference_psd.xml.gz sql
	ls *H1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *H2_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H2_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > L1_split_bank.cache
	gstlal_inspiral_iir_bank_pipe --instrument H1 --reference-psd reference_psd.xml.gz --bank-cache H1_split_bank.cache --flow $(LOW_FREQUENCY_CUTOFF) --sampleRate $(SAMPLE_RATE) --epsilon $(EPSILON) --downsample --output-name H1_bank
	gstlal_inspiral_iir_bank_pipe --instrument H2 --reference-psd reference_psd.xml.gz --bank-cache H2_split_bank.cache --flow $(LOW_FREQUENCY_CUTOFF) --sampleRate $(SAMPLE_RATE) --epsilon $(EPSILON) --downsample --output-name H2_bank
	gstlal_inspiral_iir_bank_pipe --instrument L1 --reference-psd reference_psd.xml.gz --bank-cache L1_split_bank.cache --flow $(LOW_FREQUENCY_CUTOFF) --sampleRate $(SAMPLE_RATE) --epsilon $(EPSILON) --downsample --output-name L1_bank
	cat H1_bank.dag H2_bank.dag L1_bank.dag > bank.dag

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml *split_bank-H1-TMPLTBANK-871154847-2048.xml *.xml.gz logs
