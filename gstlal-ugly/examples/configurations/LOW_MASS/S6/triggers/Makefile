# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls triggers
IFOS = H1 L1 V1
START = 968054558
STOP =  969054558
#START = 869400000
#STOP =  870600000
PSD_FILE = ../bank/reference_psd.xml.gz
BANK_CACHE_PREFIX = ../bank/
BANK_CACHE_SUFFIX = _bank.cache
# the point of this is to build the string e.g. H1=../bank/H1_bank.cache,L1=../bank/L1_bank.cache 
BANK_CACHE_STRING:= $(addprefix $(BANK_CACHE_PREFIX),$(IFOS))
BANK_CACHE_STRING:= $(addprefix =,$(BANK_CACHE_STRING))
BANK_CACHE_STRING:= $(addsuffix $(BANK_CACHE_SUFFIX),$(BANK_CACHE_STRING))
BANK_CACHE_STRING:= $(join $(IFOS),$(BANK_CACHE_STRING))
BANK_CACHE_STRING:= $(strip $(BANK_CACHE_STRING))
BANK_CACHE_STRING:= $(subst $(space),$(comma),$(BANK_CACHE_STRING))

CLUSTER_SQL = ../bank/simplify_and_cluster.sql
INJ_CLUSTER_SQL = ../bank/inj_simplify_and_cluster.sql
WEBDIR = /home/channa/public_html/BIGDOG
NUMBANKS = 4

# Segments file names
segments_suffix := _segmentspadded.xml
SEGMENTS_FILES  := $(addsuffix $(segments_suffix),$(IFOS))

TRIM = 16

# Vetoes file names
vetoes_suffix := _vetoes.xml
VETOES_FILES  := $(addsuffix $(vetoes_suffix),$(IFOS))

# Frame cache file names
frame_suffix      := _frame.cache
FRAME_CACHE_FILES := $(addsuffix $(frame_suffix),$(IFOS))

# Injections
# The seed is the string before the suffix _injections.xml
# Change as appropriate, whitespace is important
INJECTIONS := 1_injections.xml
injections:=--injections $(space)
INJECTION_LIST := $(subst $(space), $(injections), $(INJECTIONS))
BNS_WAVEFORM = TaylorT4threePointFivePN
NSBH_WAVEFORM = TaylorT4threePointFivePN
BNS_LOCAL_RATE = 1200 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
NSBH_LOCAL_RATE = 0.005 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
BNS_MAX_DIST = 100 # Mpc
NSBH_MAX_DIST = 5000 # Mpc
NS_MIN_MASS = 1.0
NS_MAX_MASS = 3.0
BH_MIN_MASS = 5.0
BH_MAX_MASS = 20.0
BH_MIN_SPIN = 0.0
BH_MAX_SPIN = 1.0
FLOW = 30


all : dag

plots :
	mkdir plots
	mkdir $(WEBDIR)

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=0:3.14159:3.14159  tisi.xml

dag :  segments.xml frame.cache tisi.xml plots $(INJECTIONS)
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache frame.cache --frame-segments-file segments.xml --frame-segments-name datasegments   --control-peak-time 4 --num-banks $(NUMBANKS) --fir-stride 4 --cluster-sql-file $(CLUSTER_SQL) --injection-sql-file $(INJ_CLUSTER_SQL) --web-dir $(WEBDIR) --time-slide-file tisi.xml $(INJECTION_LIST) --bank-cache $(BANK_CACHE_STRING) --channel=V1=h_16384Hz --channel=H1=LDAS-STRAIN --channel=L1=LDAS-STRAIN

%_vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$*:Injection --result-name=vetoes > $@
	ligolw_segments_compat $@

V1_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=V1:ITF_SCIENCEMODE --result-name=datasegments > V1segments.xml
	ligolw_segments_compat V1segments.xml
	gstlal_segments_trim V1segments.xml --trim $(TRIM) --output $@

%_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$*:DMT-SCIENCE:4 --result-name=datasegments > $*segments.xml
	ligolw_segments_compat $*segments.xml
	gstlal_segments_trim $*segments.xml --trim $(TRIM) --output $@

V1_frame.cache:
	ligo_data_find -o V -t HrecV2 -l  -s $(START) -e $(STOP) --url-type file > V1_frame.cache

%_frame.cache:
	#FIXME horrible hack to get the observatory, not guaranteed to work
	$(eval OBS:=$*)
	$(eval OBS:=$(subst 1,$(empty),$(OBS)))
	$(eval OBS:=$(subst 2,$(empty),$(OBS)))
	ligo_data_find -o $(OBS) -t $*_LDAS_C02_L2 -l  -s $(START) -e $(STOP) --url-type file > $@

frame.cache: $(FRAME_CACHE_FILES)
	cat $(FRAME_CACHE_FILES) > frame.cache

segments.xml: $(SEGMENTS_FILES)
	ligolw_add --output segments.xml $(SEGMENTS_FILES)
	ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segments.xml

%_injections.xml:
	gstlal_injections_by_local_rate \
	--seed $* \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--bns-waveform $(BNS_WAVEFORM) \
	--nsbh-waveform $(NSBH_WAVEFORM) \
	--bns-local-rate $(BNS_LOCAL_RATE) \
	--nsbh-local-rate $(NSBH_LOCAL_RATE) \
	--bns-max-distance $(BNS_MAX_DIST) \
	--nsbh-max-distance $(NSBH_MAX_DIST) \
	--ns-min-mass $(NS_MIN_MASS) \
	--ns-max-mass $(NS_MAX_MASS) \
	--bh-min-mass $(BH_MIN_MASS) \
	--bh-max-mass $(BH_MAX_MASS) \
	--bh-min-spin $(BH_MIN_SPIN) \
	--bh-max-spin $(BH_MAX_SPIN) \
	--output $@ && gstlal_injections_by_local_rate_plot $@




realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite plots $(WEBDIR) *.html Images *.css *.js *.png
