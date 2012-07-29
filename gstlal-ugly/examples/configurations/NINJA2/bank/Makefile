# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls psd segments, should set it to the analysis time intended
IFOS = H1 L1 V1
PSDSTART = 968540000
PSDSTOP =  968640000

# bank "files"
bank_suffix = _bank
BANK_FILES := $(addsuffix $(bank_suffix),$(IFOS))

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

# Template bank parameters
TEMPLATE_BANK_FILE = H1-SBANK-871154847-2048.xml.gz
OVERLAP = 2
NUM_SPLIT_TEMPLATES = 17
LOW_FREQUENCY_CUTOFF = 40
HIGH_FREQUENCY_CUTOFF = 2048

all : dag plots

%_bank: $(TEMPLATE_BANK_FILE)
	gstlal_bank_splitter --overlap $(OVERLAP) --instrument $* --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) $^ --bank-program ligolw_cbc_sbank

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

V1_psd.xml.gz : frame.cache segments.xml
	gstlal_reference_psd --channel h_16384Hz --frame-cache frame.cache --frame-segments-file segments.xml --gps-start-time $(PSDSTART) --gps-end-time $(PSDSTOP) --instrument V1 --write-psd $@ --verbose

%_psd.xml.gz : frame.cache segments.xml
	gstlal_reference_psd --channel LDAS-STRAIN --frame-cache frame.cache --frame-segments-file segments.xml --gps-start-time $(PSDSTART) --gps-end-time $(PSDSTOP) --instrument $* --write-psd $@ --verbose

reference_psd.xml.gz : $(PSD_FILES)
	ligolw_add $(PSD_FILES) --output reference_psd.xml.gz

V1_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${PSDSTART} --gps-end-time ${PSDSTOP} --include-segments=V1:ITF_SCIENCEMODE --result-name=datasegments > V1segments.xml
	ligolw_segments_compat V1segments.xml
	gstlal_segments_trim V1segments.xml --trim $(TRIM) --output $@

%_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${PSDSTART} --gps-end-time ${PSDSTOP} --include-segments=$*:DMT-SCIENCE:4 --result-name=datasegments > $*segments.xml
	ligolw_segments_compat $*segments.xml
	gstlal_segments_trim $*segments.xml --trim $(TRIM) --output $@

segments.xml: $(SEGMENTS_FILES)
	ligolw_add --output segments.xml $(SEGMENTS_FILES)
	ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segments.xml

sql :
	cat uniqueify_coinc_definer.sql uniqueify_time_slide.sql > simplify.sql
	cat simplify.sql cluster.sql > simplify_and_cluster.sql
	cat simplify.sql delete_by_far.sql > simplify_and_delete_by_far.sql
	cat drop_sim.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql

dag : $(BANK_FILES) reference_psd.xml.gz sql
	ls *H1_split_bank-H1-SBANK-871154847-2048.xml.gz | lalapps_path2cache > H1_split_bank.cache
	ls *V1_split_bank-H1-SBANK-871154847-2048.xml.gz | lalapps_path2cache > V1_split_bank.cache
	ls *L1_split_bank-H1-SBANK-871154847-2048.xml.gz | lalapps_path2cache > L1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --instrument H1 --reference-psd reference_psd.xml.gz --bank-cache H1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H1_bank
	gstlal_inspiral_svd_bank_pipe --instrument V1 --reference-psd reference_psd.xml.gz --bank-cache V1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name V1_bank
	gstlal_inspiral_svd_bank_pipe --instrument L1 --reference-psd reference_psd.xml.gz --bank-cache L1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name L1_bank
	cat V1_bank.dag H1_bank.dag L1_bank.dag > bank.dag

plots : reference_psd.xml.gz
	gstlal_plot_psd reference_psd.xml.gz psd_plot

realclean :
	rm -r *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml.gz *split_bank-H1-TMPLTBANK-871154847-2048.xml.gz mode_*.xml.gz logs *.png reference_psd.xml.gz simplify.sql  simplify_and_cluster.sql simplify_and_delete_by_far.sql inj_simplify_and_cluster.sql lalapps_path2cache *_psd.xml.gz *segments.xml *_segmentspadded.xml
