# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls triggers
IFOS = H1 L1
START = 871147452
STOP =  871747452
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
WEBDIR = /home/channa/public_html/FAR_PAPER
NUMBANKS = 2

# Segments file names
SEGMENTS = ../recolor_frames/recolor_segments.xml.gz

# Vetoes file names
vetoes_suffix := _vetoes.xml
VETOES_FILES  := $(addsuffix $(vetoes_suffix),$(IFOS))

# Frame cache file names
FRAME_CACHE := ../recolor_frames/recolor_frame.cache

# Injections
# The seed is the string before the suffix _injections.xml
# Change as appropriate, whitespace is important
INJECTIONS := 1_injections.xml
injections:=--injections $(space)
INJECTION_LIST := $(subst $(space), $(injections), $(INJECTIONS))
BNS_WAVEFORM = TaylorT4threePointFivePN
NSBH_WAVEFORM = TaylorT4threePointFivePN
BNS_LOCAL_RATE = 1200 # mergers per Mpc^3 per Myr, corresponds to "ridiculous rate"
NSBH_LOCAL_RATE = 0.005 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
BNS_MAX_DIST = 200 # Mpc
NSBH_MAX_DIST = 5000 # Mpc
NS_MIN_MASS = 1.0
NS_MAX_MASS = 2.0
BH_MIN_MASS = 5.0
BH_MAX_MASS = 20.0
BH_MIN_SPIN = 0.0
BH_MAX_SPIN = 1.0
FLOW = 10


all : dag

plots :
	mkdir plots
	mkdir $(WEBDIR)

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=L1=0:3.14159:3.14159  tisi.xml

dag :  tisi.xml plots $(INJECTIONS)
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAME_CACHE) --frame-segments-file $(SEGMENTS) --frame-segments-name datasegments   --control-peak-time 4 --num-banks $(NUMBANKS) --fir-stride 4 --cluster-sql-file $(CLUSTER_SQL) --injection-sql-file $(INJ_CLUSTER_SQL) --web-dir $(WEBDIR) --time-slide-file tisi.xml $(INJECTION_LIST) --bank-cache $(BANK_CACHE_STRING) --channel=V1=h_16384Hz --channel=H1=LSC-STRAIN --channel=L1=LSC-STRAIN

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
