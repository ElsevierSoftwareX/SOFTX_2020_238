# event at GPS 894413499.0
START = 871154700
STOP =  871156910
IFOS = H

# Template bank parameters
MIN_MASS = 3.0
MAX_MASS = 18.0
MIN_CHIRP = 7.0
MAX_CHIRP = 7.6
MIN_TOTAL_MASS = 12.0
MAX_TOTAL_MASS = 24.0
MIN_ETA = 0.13
MAX_ETA = 0.25
NUM_IN_GRID = 20
NUM_TEST_GRID = 100
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 2047.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 73
OVERLAP = 0
MM = 0.99

all : dag

frame.H.cache :
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --instrument H1 --verbose --fake-data LIGO
	ls H1-FAKE-STRAIN/*.gwf | ./lalapps_path2cache > frame.H.cache

H1-TMPLTBANK-871154847-2048.xml.gz: frame.H.cache 
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
		--frame-cache $^ \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--order twoPN \
		--verbose \
		--spectrum-type median \
		&& gzip H1-TMPLTBANK-871154847-2048.xml \
		&& ./chirp_cut H1-TMPLTBANK-871154847-2048.xml.gz $(MIN_CHIRP) $(MAX_CHIRP) \

#reference_psd.xml.gz: frame.H.cache
#	gstlal_reference_psd --channel-name FAKE-STRAIN --frame-cache frame.H.cache --instrument H1 --gps-start-time $(START) --gps-end-time $(STOP) --write-psd reference_psd.xml.gz --verbose

bank.xml.gz : H1-TMPLTBANK-871154847-2048.xml.gz
	./template_grid H1-TMPLTBANK-871154847-2048.xml.gz $(MIN_CHIRP) $(MAX_CHIRP) $(NUM_IN_GRID) $(MIN_ETA) $(MAX_ETA) $(NUM_IN_GRID) bank.xml.gz

test_bank.xml.gz : H1-TMPLTBANK-871154847-2048.xml.gz
	./template_grid H1-TMPLTBANK-871154847-2048.xml.gz $(MIN_CHIRP) $(MAX_CHIRP) $(NUM_TEST_GRID) $(MIN_ETA) $(MAX_ETA) $(NUM_TEST_GRID) test_bank.xml.gz
	
dag : H1-TMPLTBANK-871154847-2048.xml.gz bank.xml.gz test_bank.xml.gz
	ls H1-TMPLTBANK-871154847-2048.xml.gz | ./lalapps_path2cache > H1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --samples-min 4096 --samples-max-256 4096 --instrument H1 --reference-psd reference_psd.xml.gz --bank-cache H1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H1_bank

#plots : reference_psd.xml.gz
#	gstlal_plot_psd reference_psd.xml.gz psd_plot

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml.gz bank.xml.gz test_bank.xml.gz logs H1-FAKE-STRAIN
