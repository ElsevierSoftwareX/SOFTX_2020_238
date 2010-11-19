# Analaysis time
GPS_START_TIME=900000000
DURATION=1209600
GPS_END_TIME=$(shell echo $(GPS_START_TIME) + $(DURATION) | bc)


# PSD estimate and template placement time
PSD_START_TIME=$(shell echo $(GPS_START_TIME) + 1024 | bc)
PSD_END_TIME=$(shell echo $(GPS_START_TIME) + 3072 | bc)
PSD_DURATION=$(shell echo $(PSD_END_TIME) - $(PSD_START_TIME) | bc)
TMPLTBANK=H1-TMPLTBANK-$(PSD_START_TIME)-$(PSD_DURATION).xml.gz


# Mass parameters
MIN_MASS = 1.0
MAX_MASS = 3.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)


all: aligo.dag tmpltbank.xml.gz psd.xml.gz


# These targets are intermediate; it's OK to delete them if they are built only
# to satisfy a dependency for another target.
.INTERMEDIATE: $(TMPLTBANK)


aligo.dag: tmpltbank.xml.gz psd.xml.gz
	./build_job $<


# Estimate PSD
psd.xml.gz: frame.cache
	gstlal_reference_psd \
		--verbose \
		--instrument H1 \
		--channel-name MOCK-STRAIN \
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(PSD_END_TIME) \
		--frame-cache $^ \
		--write-psd $@


# Place templates
$(TMPLTBANK): frame.cache
	lalapps_tmpltbank \
		--disable-compute-moments \
		--grid-spacing Hexagonal \
		--dynamic-range-exponent 69.0 \
		--enable-high-pass 5.0 \
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
		--low-frequency-cutoff 10.0 \
		--pad-data 8 \
		--num-freq-cutoffs 1 \
		--sample-rate 4096 \
		--high-frequency-cutoff 2048.0 \
		--resample-filter ldas \
		--strain-high-pass-atten 0.1 \
		--strain-high-pass-freq 5 \
		--frame-cache $^ \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--write-compress \
		--order twoPN \
		--spectrum-type median


# Generate template bank with duplicate mass pairs pruned
tmpltbank.xml.gz: $(TMPLTBANK)
	gstlal_prune_duplicate_mass_pairs $^ $@


# Generate software injections
swinj.%.xml.gz:
	lalapps_inspinj \
		--output $@ \
		--write-compress \
		--seed $* \
		--f-lower 1	0 \
		--gps-start-time $(GPS_START_TIME) \
		--gps-end-time $(GPS_END_TIME) \
		--t-distr uniform \
		--time-step 1200 \
		--time-interval 200 \
		--i-distr uniform \
		--l-distr random \
		--d-distr uniform \
		--min-distance 1e3 \
		--max-distance 300e3 \
		--m-distr componentMass \
		--min-mass1 $(MIN_MASS) \
		--max-mass1 $(MAX_MASS) \
		--min-mass2 $(MIN_MASS) \
		--max-mass2 $(MAX_MASS) \
		--min-mtotal $(MIN_TOTAL_MASS) \
		--max-mtotal $(MAX_TOTAL_MASS) \
		--waveform GeneratePPNtwoPN \
		--disable-spin


# Generate LAL frame cache
frame.cache:
	./glob_data_find $(PWD)/hoft > $@


clean:
	rm -f frame.cache *.sub *.dag *.tmpltbank.xml.gz *.tmpltbank.xml.gz.* tmpltbank.xml.gz $(TMPLTBANK) psd.xml.gz
