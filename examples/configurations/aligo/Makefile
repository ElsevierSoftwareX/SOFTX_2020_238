# Analaysis time
GPS_START_TIME=900000000
DURATION=86400
GPS_END_TIME=$(shell echo $(GPS_START_TIME) + $(DURATION) | bc)


# PSD estimate and template placement time
PSD_START_TIME=$(shell echo $(GPS_START_TIME) + 7200 | bc)
PSD_END_TIME=$(shell echo $(PSD_START_TIME) + 2048 | bc)
PSD_DURATION=$(shell echo $(PSD_END_TIME) - $(PSD_START_TIME) | bc)
TMPLTBANK=H1-TMPLTBANK-$(PSD_START_TIME)-$(PSD_DURATION).xml


# Injection files (10)
INJECTIONS=$(foreach n,$(shell echo {0..9}),injections/injections$n.xml)


# Template bank parameters
MIN_MASS = 1.0
MAX_MASS = 3.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0


all: aligo.dag tmpltbank/tmpltbank.xml $(INJECTIONS) psd.xml


# These targets are intermediate; it's OK to delete them if they are built only
# to satisfy a dependency for another target.
.INTERMEDIATE: $(TMPLTBANK) eff_distances candle_snr min_distance max_distance


aligo.dag: tmpltbank/tmpltbank.xml psd.xml $(INJECTIONS)
	mkdir -p log
	mkdir -p svd
	mkdir -p triggers
	./build_job \
		--gps-start-time $(GPS_START_TIME) \
		--gps-end-time $(GPS_END_TIME) \
		--flow $(LOW_FREQUENCY_CUTOFF) \
		$(foreach inj,$(INJECTIONS),--injections $(inj)) \
		$<


# Generate LAL frame cache
frame.cache:
	./glob_data_find $(PWD)/hoft > $@


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
$(TMPLTBANK): frame.cache
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
		--frame-cache $^ \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--order twoPN \
		--spectrum-type median


# Generate template bank with duplicate mass pairs pruned
tmpltbank/tmpltbank.xml: $(TMPLTBANK)
	mkdir -p $(@D)
	gstlal_prune_duplicate_mass_pairs $^ $@


# Read and sort effective distances for entire template bank.
# FIXME: masses to scan from summ_value table are hard coded!
eff_distances: tmpltbank/tmpltbank.xml
	ligolw_print -t summ_value -c name -c comment -c value $< \
	| sed -n s/inspiral_effective_distance,//p \
	| sed -n "s/[1-3]\.00_[1-3]\.00.*,//p" \
	| sort -n > $@


# Read candle SNR argument from template bank
candle_snr: tmpltbank/tmpltbank.xml
	ligolw_print --table process_params --column param --column value $< \
	| sed -n s/--candle-snr,//p > $@


# Determine minimum distance for injections
min_distance: eff_distances candle_snr
	echo "print 0.5 * $(shell cat candle_snr) / 8.0 * $(shell head -n 1 eff_distances) * 1e3" | python > $@


# Determine maximum distance for injections
max_distance: eff_distances candle_snr
	echo "print 0.25 * $(shell cat candle_snr) / 5.5 * $(shell tail -n 1 eff_distances) * 1e3" | python > $@


# Generate software injections
injections/injections%.xml: min_distance max_distance
	mkdir -p $(@D)
	lalapps_inspinj \
		--output $@ \
		--seed $* \
		--f-lower $(LOW_FREQUENCY_CUTOFF) \
		--gps-start-time $(GPS_START_TIME) \
		--gps-end-time $(GPS_END_TIME) \
		--t-distr uniform \
		--time-step 80 \
		--time-interval 20 \
		--i-distr uniform \
		--l-distr random \
		--d-distr log10 \
		--min-distance $(shell cat min_distance) \
		--max-distance $(shell cat max_distance) \
		--m-distr componentMass \
		--min-mass1 $(MIN_MASS) \
		--max-mass1 $(MAX_MASS) \
		--min-mass2 $(MIN_MASS) \
		--max-mass2 $(MAX_MASS) \
		--min-mtotal $(MIN_TOTAL_MASS) \
		--max-mtotal $(MAX_TOTAL_MASS) \
		--waveform GeneratePPNtwoPN \
		--disable-spin \
		--taper-injection start


clean:
	rm -f \
		eff_distances candle_snr min_distance max_distance \
		frame.cache \
		psd.xml \
		$(TMPLTBANK) \
		*.{sub,dag} \
		*.dag.*
	rm -rf tmpltbank log svd injections triggers
