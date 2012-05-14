START = 871147552
STOP = 871752352 
#STOP = 876357464 
PSD_FILE = ../psd/reference_psd.xml.gz
DELETE_SQL_FILE = simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = simplify_and_cluster.sql
INJ_SQL_FILE = inj_simplify_and_cluster.sql
H1_BANK_CACHE = ../bank/H1_svd_bank.cache
L1_BANK_CACHE = ../bank/L1_svd_bank.cache
V1_BANK_CACHE = ../bank/V1_svd_bank.cache
FRAMES = ../frames/frames.cache
WEBDIR = /home/mafrei/public_html/ #Please change to your public_html dir
SEGMENTS_FILE = ../psd/segments.xml.gz

all : dag

tisi.xml:
	ligolw_tisi --comment "NINJA2MONTH" -i H1=0:0:0 -i L1=0:0:0 -i V1=0:0:0 tisi.xml

dag : web tisi.xml
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAMES) --frame-segments-file ${SEGMENTS_FILE} --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --cluster-sql-file $(CLUSTER_SQL_FILE) --injection-sql-file $(INJ_SQL_FILE) --web-dir $(WEBDIR) --time-slide-file tisi.xml --channel H1=LDAS-STRAIN --channel L1=LDAS-STRAIN --channel V1=h_16384Hz --vetoes vetoes.xml.gz --control-peak-time 4 --fir-stride 4 --coincidence-threshold 0.030
	#sed --in-place 's/vanilla/local/' gstlal_inspiral.sub
	sed --in-place 's/request_cpus = 6//' gstlal_inspiral.sub 
	sed --in-place 's/requirements = ( CAN_RUN_MULTICORE )//' gstlal_inspiral.sub
	sed --in-place 's/+RequiresMultipleCores = True/requirements = Memory > 3000/' gstlal_inspiral.sub
	sed --in-place -i '5 d' gstlal_inspiral.sub
	sed --in-place -i '4 d' gstlal_inspiral.sub
	sed --in-place '$$i\+Online_CBC_GSTLAL=True' gstlal_inspiral.sub

web :
	mkdir plots
	mkdir -p  $(WEBDIR) 

realclean :
	rm -r plots $(WEBDIR) *.sub *.dag* *.cache *.sh logs *.xml segments.xml.gz segments.txt *.sqlite H*.xml.gz
