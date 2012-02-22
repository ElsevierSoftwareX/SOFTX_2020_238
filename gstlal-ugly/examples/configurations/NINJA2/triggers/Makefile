START = 871147552
STOP = 871752352 
#STOP = 876357464 
PSD_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//psd/reference_psd.xml.gz
DELETE_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/simplify_and_cluster.sql
INJ_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/inj_simplify_and_cluster.sql
H1_BANK_CACHE = ../../H1_svd_bank.cache
L1_BANK_CACHE = ../../L1_svd_bank.cache
V1_BANK_CACHE = ../../V1_svd_bank.cache
FRAMES = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//frames/frames.cache
WEBDIR = /home/mafrei/public_html/NINJA2/TwoMonthSet/sbank
SEGMENTS_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format/psd/segments.xml.gz

all : dag

tisi.xml:
	ligolw_tisi --comment "NINJA2MONTH" -i H1=0:0:0 -i L1=0:0:0 -i V1=0:0:0 tisi.xml

dag : web tisi.xml
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAMES) --frame-segments-file ${SEGMENTS_FILE} --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --delete-sql-file $(DELETE_SQL_FILE) --cluster-sql-file $(CLUSTER_SQL_FILE) --injection-sql-file $(INJ_SQL_FILE) --web-dir $(WEBDIR) --time-slide-file tisi.xml --channel H1=LDAS-STRAIN --channel L1=LDAS-STRAIN --channel V1=h_16384Hz --vetoes vetoes.xml.gz --control-peak-time 4 --fir-stride 4

web :
	mkdir plots
	mkdir -p  $(WEBDIR) 

realclean :
	rm -r plots $(WEBDIR) *.sub *.dag* *.cache *.sh logs *.xml segments.xml.gz segments.txt *.sqlite H*.xml.gz
