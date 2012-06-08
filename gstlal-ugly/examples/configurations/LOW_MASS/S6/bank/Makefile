# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls psd segments, should set it to the analysis time intended
IFOS = H1 L1 V1
PSDSTART = 968054558
PSDSTOP =  969054558

# Segments file names
segments_suffix := _segmentspadded.xml
SEGMENTS_FILES  := $(addsuffix $(segments_suffix),$(IFOS))


# reference psd file names
psd_suffix := _psd.xml.gz
PSD_FILES  := $(addsuffix $(psd_suffix),$(IFOS))

TRIM = 16

# Frame cache file names
frame_suffix      := _frame.cache
FRAME_CACHE_FILES := $(addsuffix $(frame_suffix),$(IFOS))

# data for lalapps_tmpltbank, note this is just to be able to run the program, the psd is replaced withe the design psd
START = 871147452
STOP =  871247452
IFOS = H

# Template bank parameters
MIN_MASS = 1.0
MAX_MASS = 10.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 2047.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 250
OVERLAP = 50
MM = 0.99

all : dag plots

frame.H.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H.cache

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
		&& gzip H1-TMPLTBANK-871154847-2048.xml \
		&& gstlal_bank_splitter --overlap $(OVERLAP) --instrument H1 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml.gz \
		&& gstlal_bank_splitter --overlap $(OVERLAP) --instrument V1 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml.gz \
		&& gstlal_bank_splitter --overlap $(OVERLAP) --instrument L1 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) H1-TMPLTBANK-871154847-2048.xml.gz

reference_psd.xml.gz : $(PSD_FILES)
	ligolw_add $(PSD_FILES) --output reference_psd.xml.gz

V1_psd.xml.gz : frame.cache segments.xml
	gstlal_reference_psd --channel h_16384Hz --frame-cache frame.cache --frame-segments-file segments.xml --gps-start-time $(PSDSTART) --gps-end-time $(PSDSTOP) --instrument V1 --write-psd $@ --verbose

%_psd.xml.gz : frame.cache segments.xml
	gstlal_reference_psd --channel LDAS-STRAIN --frame-cache frame.cache --frame-segments-file segments.xml --gps-start-time $(PSDSTART) --gps-end-time $(PSDSTOP) --instrument $* --write-psd $@ --verbose
	

V1_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${PSDSTART} --gps-end-time ${PSDSTOP} --include-segments=V1:ITF_SCIENCEMODE --result-name=datasegments > V1segments.xml
	ligolw_segments_compat V1segments.xml
	gstlal_segments_trim V1segments.xml --trim $(TRIM) --output $@

%_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${PSDSTART} --gps-end-time ${PSDSTOP} --include-segments=$*:DMT-SCIENCE:4 --result-name=datasegments > $*segments.xml
	ligolw_segments_compat $*segments.xml
	gstlal_segments_trim $*segments.xml --trim $(TRIM) --output $@

V1_frame.cache:
	ligo_data_find -o V -t HrecV2 -l  -s $(PSDSTART) -e $(PSDSTOP) --url-type file > V1_frame.cache

%_frame.cache:
	#FIXME horrible hack to get the observatory, not guaranteed to work
	$(eval OBS:=$*)
	$(eval OBS:=$(subst 1,$(empty),$(OBS)))
	$(eval OBS:=$(subst 2,$(empty),$(OBS)))
	ligo_data_find -o $(OBS) -t $*_LDAS_C02_L2 -l  -s $(PSDSTART) -e $(PSDSTOP) --url-type file > $@

frame.cache: $(FRAME_CACHE_FILES)
	cat $(FRAME_CACHE_FILES) > frame.cache

segments.xml: $(SEGMENTS_FILES)
	ligolw_add --output segments.xml $(SEGMENTS_FILES)
	ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segments.xml

sql :
	cat uniqueify_coinc_definer.sql uniqueify_time_slide.sql > simplify.sql
	cat simplify.sql cluster.sql > simplify_and_cluster.sql
	cat simplify.sql delete_by_far.sql > simplify_and_delete_by_far.sql
	cat drop_sim.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql

dag : H1-TMPLTBANK-871154847-2048.xml.gz reference_psd.xml.gz sql
	ls *H1_split_bank-H1-TMPLTBANK-871154847-2048.xml.gz | lalapps_path2cache > H1_split_bank.cache
	ls *V1_split_bank-H1-TMPLTBANK-871154847-2048.xml.gz | lalapps_path2cache > V1_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871154847-2048.xml.gz | lalapps_path2cache > L1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --instrument H1 --reference-psd reference_psd.xml.gz --bank-cache H1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H1_bank
	gstlal_inspiral_svd_bank_pipe --instrument V1 --reference-psd reference_psd.xml.gz --bank-cache V1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name V1_bank
	gstlal_inspiral_svd_bank_pipe --instrument L1 --reference-psd reference_psd.xml.gz --bank-cache L1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name L1_bank
	cat V1_bank.dag H1_bank.dag L1_bank.dag > bank.dag

plots : reference_psd.xml.gz
	gstlal_plot_psd reference_psd.xml.gz psd_plot

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml.gz *split_bank-H1-TMPLTBANK-871154847-2048.xml.gz mode_*.xml.gz logs *.png reference_psd.xml.gz simplify.sql  simplify_and_cluster.sql simplify_and_delete_by_far.sql inj_simplify_and_cluster.sql lalapps_path2cache *_psd.xml.gz *segments.xml *_segmentspadded.xml
