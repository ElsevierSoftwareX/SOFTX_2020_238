START = 871147452
STOP = 871247452
PSD_FILE = ../bank/reference_psd.xml
DELETE_SQL_FILE = ../bank/simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = ../bank/simplify_and_cluster.sql
INJ_SQL_FILE = ../bank/inj_simplify_and_cluster.sql
H1_BANK_CACHE = ../bank/H1_reference_psd_svd_bank.cache
L1_BANK_CACHE = ../bank/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = ../bank/V1_reference_psd_svd_bank.cache
TISI = ../bank/tisi.xml
FRAMES = ../frames/frames.cache
SEED = 1
WEBDIR = /home/channa/public_html/ALIGO-2011.08.31


LOCAL_RATE = 2000.0 # mergers per Mpc^3 per Myr, 12 corresponds to "optimistic rate"
MAX_DIST = 200 # Mpc
SEED = 1
FLOW = 14

all : injections

vetoes.xml.gz :
	ligolw_segments --verbose --output vetoes.xml.gz --name vetoes --insert-from-segwizard=H1=/dev/null --insert-from-segwizard=L1=/dev/null --insert-from-segwizard=V1=/dev/null

segments.xml.gz :
	echo 0 $(START) $(STOP) $(shell echo $(STOP)-$(START)  | bc) > segments.txt
	ligolw_segments --verbose --output segments.xml.gz --name datasegments --insert-from-segwizard=H1=segments.txt --insert-from-segwizard=L1=segments.txt --insert-from-segwizard=V1=segments.txt

injections : injections.xml web vetoes.xml.gz segments.xml.gz
	gstlal_s5_pbh_trigger_pipe --injections injections.xml --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAMES) --frame-segments-file segments.xml.gz --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --delete-sql-file $(DELETE_SQL_FILE) --cluster-sql-file $(CLUSTER_SQL_FILE) --injection-sql-file $(INJ_SQL_FILE) --web-dir $(WEBDIR) --time-slide-file $(TISI) --vetoes vetoes.xml.gz

injections.xml:
	gstlal_injections_by_local_rate \
	--seed $(SEED) \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--local-rate $(LOCAL_RATE) \
	--output injections.xml \
	--max-distance $(MAX_DIST)

web :
	mkdir plots
	mkdir $(WEBDIR) 

realclean :
	rm -r plots $(WEBDIR) *.sub *.dag* *.cache *.sh logs *.xml vetoes.xml.gz segments.xml.gz segments.txt
