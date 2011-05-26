# PSD estimate and template placement time
DATA_START_TIME=900000000
PSD_START_TIME=$(shell echo $(DATA_START_TIME) + 16 | bc)
PSD_END_TIME=$(shell echo $(PSD_START_TIME) + 2048 | bc)
DATA_END_TIME=$(shell echo $(PSD_END_TIME) + 16 | bc)
PSD_DURATION=$(shell echo $(PSD_END_TIME) - $(PSD_START_TIME) | bc)
TMPLTBANK=H1-TMPLTBANK-$(PSD_START_TIME)-$(PSD_DURATION).xml


# Template bank parameters
MIN_MASS = 1.32
MAX_MASS = 1.4
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 15.0
HIGH_PASS_FREQ = 10.0


# Fractional part of SVD tolerances we are interested in
TOLERANCES = 9 99 999 9999 99999


all: $(foreach x,$(TOLERANCES),svd_0_$x.xml match_0_$x_0.out) tmpltbank.bin tmpltbank.xml psd.xml


# These targets are intermediate; it's OK to delete them if they are built only
# to satisfy a dependency for another target.
.INTERMEDIATE: $(TMPLTBANK)


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
tmpltbank.xml: $(TMPLTBANK)
	mkdir -p $(@D)
	gstlal_prune_duplicate_mass_pairs $^ $@


# Realize template bank as FIR filter coefficents saved to flat binary file
tmpltbank.bin: tmpltbank.xml psd.xml
	./generate_templates \
		--flow $(LOW_FREQUENCY_CUTOFF) \
		--template-bank tmpltbank.xml \
		--reference-psd psd.xml \
		--output $@


svd_0_%.xml: tmpltbank.xml psd.xml
	gstlal_svd_bank \
		--flow $(LOW_FREQUENCY_CUTOFF) \
		--svd-tolerance 0.$* \
		--template-bank tmpltbank.xml \
		--reference-psd psd.xml \
		--write-svd-bank $@ \
		--verbose


match_0_%_0.out: svd_0_%.xml tmpltbank.bin libcomputematchsink.so
	./compute_match \
		--templates tmpltbank.bin \
		--svd-bank $< \
		--output match_0_$*_%d.out


computematchsink.o: computematchsink.c
	gcc -fPIC $(shell pkg-config --cflags gstreamer-0.10) -c $< -o $@


libcomputematchsink.so: computematchsink.o
	gcc -shared $< $(shell pkg-config --libs gstreamer-0.10) -o $@


clean:
	rm -f \
		frame.cache \
		psd.xml \
		$(TMPLTBANK) \
		tmpltbank.xml \
		tmpltbank.bin \
		svd_0_*.xml \
		match_0_*_*.out \
		computematchsink.o \
		libcomputematchsink.so
	rm -rf hoft
