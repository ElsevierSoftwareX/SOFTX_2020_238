START = 871147452
STOP = 871247452
H1_BANK_CACHE = ../bank/H1_reference_psd_svd_bank.cache
H2_BANK_CACHE = ../bank/H1_reference_psd_svd_bank.cache
L1_BANK_CACHE = ../bank/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = ../bank/V1_reference_psd_svd_bank.cache

SEED = 1

LOCAL_RATE = 200.0 # mergers per Mpc^3 per Myr, 12 corresponds to "optimistic rate"
MAX_DIST = 200 # Mpc
SEED = 1
FLOW = 14

all : injections

injections : injections.xml
	gstlal_ll_trigger_pipe --injections injections.xml --gps-start-time $(START) --gps-stop-time $(STOP) --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) 

injections.xml:
	gstlal_injections_by_local_rate \
	--seed $(SEED) \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--local-rate $(LOCAL_RATE) \
	--output injections.xml \
	--max-distance $(MAX_DIST)

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml
