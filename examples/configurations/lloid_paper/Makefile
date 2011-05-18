# PSD estimate and template placement time
DATA_START_TIME=900000000
PSD_START_TIME=$(shell echo $(DATA_START_TIME) + 16 | bc)
PSD_END_TIME=$(shell echo $(PSD_START_TIME) + 2048 | bc)
DATA_END_TIME=$(shell echo $(PSD_END_TIME) + 16 | bc)
PSD_DURATION=$(shell echo $(PSD_END_TIME) - $(PSD_START_TIME) | bc)
TMPLTBANK=H1-TMPLTBANK-$(PSD_START_TIME)-$(PSD_DURATION).xml


# Template bank parameters
MIN_MASS = 1.0
MAX_MASS = 3.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 15.0
HIGH_PASS_FREQ = 10.0


all: lloid_paper.dag tmpltbank/tmpltbank.xml psd.xml


# These targets are intermediate; it's OK to delete them if they are built only
# to satisfy a dependency for another target.
.INTERMEDIATE: $(TMPLTBANK)


lloid_paper.dag: tmpltbank/tmpltbank.xml psd.xml $(INJECTIONS)
	mkdir -p log
	mkdir -p svd
	mkdir -p match
	./build_job \
		--flow $(LOW_FREQUENCY_CUTOFF) \
		$<


hoft:
	mkdir -p hoft && cd hoft && \
	gstlal_mock_data_server -i H1 -s $(DATA_START_TIME) -e $(DATA_END_TIME)


# Generate LAL frame cache
frame.cache: hoft
	find $< -name *.gwf | lalapps_path2cache > $@


# Estimate PSD
psd.xml: frame.cache
	gstlal_reference_psd \
		--instrument H1 \
		--channel-name MOCK-STRAIN \
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(PSD_END_TIME) \
		--frame-cache $^ \
		--write-psd $@


# Place templates
$(TMPLTBANK): frame.cache hoft
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
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(PSD_END_TIME) \
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1 \
		--candle-mass2 1 \
		--channel-name H1:MOCK-STRAIN \
		--space Tau0Tau3 \
		--number-of-segments 127 \
		--minimal-match 0.97 \
		--candle-snr 8 \
		--debug-level 33 \
		--high-pass-attenuation 0.1 \
		--min-high-freq-cutoff SchwarzISCO \
		--segment-length 131072 \
		--low-frequency-cutoff $(LOW_FREQUENCY_CUTOFF) \
		--pad-data 8 \
		--num-freq-cutoffs 1 \
		--sample-rate 4096 \
		--high-frequency-cutoff 2048.0 \
		--resample-filter ldas \
		--strain-high-pass-atten 0.1 \
		--strain-high-pass-freq $(HIGH_PASS_FREQ) \
		--frame-cache $< \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--order twoPN \
		--spectrum-type median


# Generate template bank with duplicate mass pairs pruned
tmpltbank/tmpltbank.xml: $(TMPLTBANK)
	mkdir -p $(@D)
	gstlal_prune_duplicate_mass_pairs $^ $@


clean:
	rm -f \
		frame.cache \
		psd.xml \
		$(TMPLTBANK) \
		*.{sub,dag} \
		*.dag.*
	rm -rf tmpltbank log svd hoft match
