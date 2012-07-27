# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls triggers
IFOS = H1 L1
START = 871147452
STOP =  871247452
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
WEBDIR = /home/channa/public_html/PBH
FRAME_CACHE = ../bank/frame.cache
SEGMENTS = ../bank/segments.xml

NUMBANKS = 2

# Vetoes file names
vetoes_suffix := _vetoes.xml
VETOES_FILES  := $(addsuffix $(vetoes_suffix),$(IFOS))

# Injections
# The seed is the string before the suffix _injections.xml
# Change as appropriate, whitespace is important
INJECTIONS := 1_injections.xml
injections:=--injections $(space)
INJECTION_LIST := $(subst $(space), $(injections), $(INJECTIONS))
MIN_DIST = 100
MAX_DIST = 10000
MIN_MASS1 = 0.25
MAX_MASS1 = 2.0
MIN_MASS2 = 0.25
MAX_MASS2 = 2.0
MIN_TOTAL_MASS = 0.5
MAX_TOTAL_MASS = 4.0
FLOW = 39
TRIM = 16

all : dag

plots :
	mkdir plots
	mkdir $(WEBDIR)

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=L1=0:3.14159:3.14159 tisi.xml

dag : vetoes.xml tisi.xml plots $(INJECTIONS)
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAME_CACHE) --frame-segments-file $(SEGMENTS) --frame-segments-name datasegments  --vetoes vetoes.xml --control-peak-time 4 --num-banks $(NUMBANKS) --fir-stride 4 --cluster-sql-file $(CLUSTER_SQL) --injection-sql-file $(INJ_CLUSTER_SQL) --web-dir $(WEBDIR) --time-slide-file tisi.xml $(INJECTION_LIST) --bank-cache $(BANK_CACHE_STRING)

%_vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$*:Injection --result-name=vetoes > $@
	ligolw_segments_compat $@

vetoes.xml: $(VETOES_FILES)
	ligolw_add $(VETOES_FILES) > vetoes.xml

%_injections.xml: 
	lalapps_inspinj \
		--output $@ \
		--seed $* \
		--f-lower $(FLOW) \
		--gps-start-time $(START) \
		--gps-end-time $(STOP) \
		--t-distr uniform \
		--time-step 100 \
		--time-interval 20 \
		--i-distr uniform \
		--l-distr random \
		--d-distr log10 \
		--min-distance $(MIN_DIST) \
		--max-distance $(MAX_DIST) \
		--m-distr componentMass \
		--min-mass1 $(MIN_MASS1) \
		--max-mass1 $(MAX_MASS1) \
		--min-mass2 $(MIN_MASS2) \
		--max-mass2 $(MAX_MASS2) \
		--min-mtotal $(MIN_TOTAL_MASS) \
		--max-mtotal $(MAX_TOTAL_MASS) \
		--waveform  TaylorT4threePointFivePN \
		--taper-injection start \
		--disable-spin

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite plots $(WEBDIR) *.html Images *.css *.js
