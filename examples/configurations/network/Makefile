PSD_START_TIME=969333979
PSD_END_TIME=969335142
FRAME_FILE_TYPE=$(if $(filter H1,$(1)),H1_DMT_C00_L2,$(if $(filter L1,$(1)),L1_DMT_C00_L2,$(if $(filter V1,$(1)),HrecOnline,)))
CHANNEL_NAME=$(if $(filter H1,$(1)),DMT-STRAIN,$(if $(filter L1,$(1)),DMT-STRAIN,$(if $(filter V1,$(1)),h_16384Hz,)))

FILES = \
	H1-TMPLTBANK_LLOID_S6_ONLINE-$(PSD_START_TIME)-2048.xml.gz \
	tmpltbank.xml.gz \
	reference_psd.H1.xml.gz \
	reference_psd.L1.xml.gz \
	reference_psd.V1.xml.gz \
	bank.H1.pickle \
	bank.L1.pickle \
	bank.V1.pickle \
	swinj.xml

MIN_MASS = 1.2
MAX_MASS = 1.6
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)

all: $(FILES)

.PHONY: clean
clean:
	rm -f $(FILES)

# Injections go from July through October 1 00:00 UTC
swinj.xml:
	lalapps_inspinj \
		--output $@ \
		--user-tag LLOID_S6_ONLINE \
		--gps-start-time 961977615 \
		--gps-end-time 969926415 \
		--t-distr fixed \
		--time-step 300 \
		--d-distr volume \
		--min-distance 1e3 \
		--max-distance 5e3 \
		--l-distr random \
		--m-distr componentMass \
		--min-mass1 $(MIN_MASS) \
		--max-mass1 $(MAX_MASS) \
		--min-mass2 $(MIN_MASS) \
		--max-mass2 $(MAX_MASS) \
		--min-mtotal $(MIN_TOTAL_MASS) \
		--max-mtotal $(MAX_TOTAL_MASS) \
		--i-distr uniform \
		--waveform GeneratePPNtwoPN \
		--f-lower 10.0 \
		--disable-spin \
		--verbose

ligo_data_find.%1.out:
	ligo_data_find \
		--observatory $* \
		--type $(call FRAME_FILE_TYPE,$*1) \
		--url-type file \
		--lal-cache \
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(shell echo $(PSD_START_TIME) + 3072 | bc) > $@

H1-TMPLTBANK_LLOID_S6_ONLINE-$(PSD_START_TIME)-2048.xml.gz: ligo_data_find.H1.out
	lalapps_tmpltbank \
		--verbose \
		--disable-compute-moments \
		--user-tag LLOID_S6_ONLINE \
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(shell echo $(PSD_START_TIME) + 2048 | bc) \
		--grid-spacing Hexagonal \
		--dynamic-range-exponent 69.0 \
		--enable-high-pass 30.0 --high-pass-order 8 \
		--strain-high-pass-order 8 \
		--minimum-mass $(MIN_MASS) --maximum-mass $(MAX_MASS) \
		--approximant TaylorF2 --order twoPN \
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1.4 --candle-mass2 1.4 \
		--channel-name H1:DMT-STRAIN --frame-cache $^ \
		--space Tau0Tau3 --number-of-segments 15 \
		--minimal-match 0.98 --candle-snr 8 --debug-level 33 --high-pass-attenuation 0.1 \
		--min-high-freq-cutoff SchwarzISCO --max-high-freq-cutoff SchwarzISCO \
		--segment-length 1048576 \
		--low-frequency-cutoff 40.0 --pad-data 8 --num-freq-cutoffs 1 \
		--sample-rate 4096 --high-frequency-cutoff 1843.2 --resample-filter ldas \
		--strain-high-pass-atten 0.1 --strain-high-pass-freq 30 \
		--min-total-mass $(MIN_TOTAL_MASS) --max-total-mass $(MAX_TOTAL_MASS) \
		--write-compress --spectrum-type median

tmpltbank.xml.gz: H1-TMPLTBANK_LLOID_S6_ONLINE-$(PSD_START_TIME)-2048.xml.gz
	gstlal_prune_duplicate_mass_pairs $< $@

reference_psd.%.xml.gz: ligo_data_find.%.out
	gstlal_reference_psd \
		--verbose \
		--gps-start-time $(PSD_START_TIME) \
		--gps-end-time $(PSD_END_TIME) \
		--channel-name $(call CHANNEL_NAME,$*) \
		--frame-cache $^ \
		--instrument $* \
		--write-psd $@

bank.%.pickle: reference_psd.%.xml.gz tmpltbank.xml.gz
	gstlal_svd_bank \
		--verbose \
		--template-bank tmpltbank.xml.gz \
		--ortho-gate-fap 0.01 \
		--reference-psd $< \
		--write-bank $@


#
# Segment queries
#

# Definitions
VETODEF=H1L1V1-S6_CBC_LOWMASS_D_PSEUDO_ONLINE.xml
VETODEFURL=https://www.lsc-group.phys.uwm.edu/ligovirgo/cbc/public/segments/S6/S6D/$(VETODEF)
VETOSTARTTIME=$(shell lalapps_tconvert September 24, 2010)
VETOENDTIME=$(shell lalapps_tconvert October 1, 2010)
VETODURATION=$(shell echo $(VETOENDTIME) - $(VETOSTARTTIME) | bc)
IFOS=H1 L1 V1
CATS=1 2 3 4 5
SCIENCE_SEGMENTS=H1:DMT-SCIENCE:1 L1:DMT-SCIENCE:1 V1:ITF_SCIENCEMODE:1

# Download veto definer
$(VETODEF):
	wget --no-check-certificate $(VETODEFURL)

# Query segment database for cumulative veto times for all categories
$(foreach cat,$(CATS),$(foreach ifo,$(IFOS),$(ifo)-VETOTIME_CAT$(cat)-$(VETOSTARTTIME)-$(VETODURATION).xml)):
	ligolw_segments_from_cats \
		--database \
		--veto-file $(VETODEF) \
		--cumulative-categories \
		--gps-start-time $(VETOSTARTTIME) \
		--gps-end-time $(VETOENDTIME)

# Query segment database for science segments
# Arrghh! ligolw_cut needed because of incompatible columns!
science_segments.%.xml:
	ligolw_segment_query \
		--database \
		--query-segments \
		--include-segments $(filter $(*):%,$(SCIENCE_SEGMENTS)) \
		--gps-start-time $(VETOSTARTTIME) \
		--gps-end-time $(VETOENDTIME) \
		--result-name SCIENCE \
		-o $@
	ligolw_cut \
		--delete-column process:domain \
		--delete-column process:is_online \
		--delete-column process:jobid \
		$@

# Subtract vetoed segments from science segments
# Arrghh! ligolw_segment_diff does not return exit code 0 iff successful!
science_cat3_segments.%.xml: science_segments.%.xml %-VETOTIME_CAT3-$(VETOSTARTTIME)-$(VETODURATION).xml
	ligolw_segment_diff \
		--segment \
		--include-segments $*:SCIENCE,$*:VETO_CAT3_CUMULATIVE \
		--result-name $*_SCIENCE_CAT3 \
		-o $@ $^ || test -e $@

# Intersect all cat5 science segments from all detectors
# Arrghh! ligolw_segment_intersect does not return exit code 0 iff successful!
intersection.xml: $(foreach ifo,$(IFOS),science_cat3_segments.$(ifo).xml)
	ligolw_segment_intersect \
		--segment \
		--include-segments :H1_SCIENCE_CAT3,:L1_SCIENCE_CAT3,:V1_SCIENCE_CAT3 \
		-o $@ $^ || test -e $@

intersection_spans.txt: intersection.xml
	ligolw_print \
		--table segment \
		--column end_time \
		--column start_time \
		--delimiter "-" \
		$^ | bc > $@

intersection_start_times.txt: intersection.xml
	ligolw_print \
		--table segment \
		--column start_time \
		$^ > $@

intersection_end_times.txt: intersection.xml
	ligolw_print \
		--table segment \
		--column end_time \
		$^ > $@

longest_triple_time.txt: intersection_spans.txt intersection_start_times.txt intersection_end_times.txt
	echo "# Duration  Start Time  End Time" > $@
	paste $^ | sort -nr >> $@
