START = 871147452
STOP = 871197452
PSD_FILE = ../bank2/reference_psd.xml
DELETE_SQL_FILE = ../bank2/simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = ../bank2/simplify_and_cluster.sql
INJ_SQL_FILE = ../bank2/inj_simplify_and_cluster.sql
H1_BANK_CACHE = ../bank2/H1_reference_psd_svd_bank.cache
L1_BANK_CACHE = ../bank2/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = ../bank2/V1_reference_psd_svd_bank.cache
TISI = ../bank2/tisi.xml
FRAMES = ../frames/frames.cache
SEED = 1
WEBDIR = /home/channa/public_html/ALIGO-2011.08.15

MIN_DIST = 10000
MAX_DIST = 200000
MIN_MASS1 = 1.0
MAX_MASS1 = 3.0
MIN_MASS2 = 1.0
MAX_MASS2 = 3.0
MIN_TOTAL_MASS = 2.0
MAX_TOTAL_MASS = 6.0
FLOW = 14
TRIM = 16

all : injections

vetoes.xml.gz :
	ligolw_segments --verbose --output vetoes.xml.gz --name vetoes --insert-from-segwizard=H1=/dev/null --insert-from-segwizard=L1=/dev/null --insert-from-segwizard=V1=/dev/null

segments.xml.gz :
	echo 0 $(START) $(STOP) $(shell echo $(STOP)-$(START)  | bc) > segments.txt
	ligolw_segments --verbose --output segments.xml.gz --name datasegments --insert-from-segwizard=H1=segments.txt --insert-from-segwizard=L1=segments.txt --insert-from-segwizard=V1=segments.txt

injections : injections.xml web vetoes.xml.gz segments.xml.gz
	gstlal_s5_pbh_trigger_pipe --injections injections.xml --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAMES) --frame-segments-file segments.xml.gz --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --delete-sql-file $(DELETE_SQL_FILE) --cluster-sql-file $(CLUSTER_SQL_FILE) --injection-sql-file $(INJ_SQL_FILE) --web-dir $(WEBDIR) --time-slide-file $(TISI) --vetoes vetoes.xml.gz

injections.xml:
	lalapps_inspinj \
		--output injections.xml \
		--seed $(SEED) \
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
		--waveform GeneratePPNthreePointFivePN \
		--taper-injection start \
		--disable-spin

web :
	mkdir plots
	mkdir $(WEBDIR) 

realclean :
	rm -r plots $(WEBDIR) *.sub *.dag* *.cache *.sh logs *.xml vetoes.xml.gz segments.xml.gz segments.txt
