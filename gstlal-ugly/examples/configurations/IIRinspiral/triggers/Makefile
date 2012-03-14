START = 869622409
STOP = 869642409
PSD_FILE = /home/shooper/scratch/IIR/PBH/bank/reference_psd.xml.gz
DELETE_SQL_FILE = /home/shooper/scratch/IIR/PBH/bank/simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = /home/shooper/scratch/IIR/PBH/bank/simplify_and_cluster.sql
INJ_SQL_FILE = /home/shooper/scratch/IIR/PBH/bank/inj_simplify_and_cluster.sql
H1_BANK_CACHE = /home/shooper/scratch/IIR/PBH/bank/mode_H1-TMPLTBANK-869040003-2048_iir_bank.cache
H2_BANK_CACHE = /home/shooper/scratch/IIR/PBH/bank/mode_H2-TMPLTBANK-875211248-2048_iir_bank.cache
L1_BANK_CACHE = /home/shooper/scratch/IIR/PBH/bank/mode_L1-TMPLTBANK-858087370-2048_iir_bank.cache
SEED = 1
WEBDIR = /home/shooper/public_html/sub-solar-mass-869622409-869642409

MIN_DIST = 770
MAX_DIST = 10000
MIN_MASS1 = 1.2
MAX_MASS1 = 1.6
MIN_MASS2 = 1.2
MAX_MASS2 = 1.6
MIN_TOTAL_MASS = 2.4
MAX_TOTAL_MASS = 3.4
FLOW = 39
TRIM = 16

all : injections

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=0:3.14159:3.14159 tisi.xml

dag : vetoes.xml segments.xml frame.cache tisi.xml web
	gstlal_s5_pbh_trigger_pipe --do-iir-pipeline --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache frame.cache --frame-segments-file segments.xml  --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes vetoes.xml --time-slide-file tisi.xml --delete-sql-file $(DELETE_SQL_FILE) --cluster-sql-file $(CLUSTER_SQL_FILE) --web-dir $(WEBDIR) --control-peak-time 8

injections : vetoes.xml segments.xml frame.cache injections.xml web tisi.xml
	gstlal_s5_pbh_trigger_pipe --do-iir-pipeline --injections injections.xml --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache frame.cache --frame-segments-file segments.xml --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes vetoes.xml --cluster-sql-file $(CLUSTER_SQL_FILE) --injection-sql-file $(INJ_SQL_FILE) --web-dir $(WEBDIR) --time-slide-file tisi.xml  --control-peak-time 8

H1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Injection --result-name=vetoes > H1vetoes.xml
	ligolw_segments_compat H1vetoes.xml

H2vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Injection --result-name=vetoes > H2vetoes.xml
	ligolw_segments_compat H2vetoes.xml

L1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Injection --result-name=vetoes > L1vetoes.xml
	ligolw_segments_compat L1vetoes.xml

H1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  --result-name=datasegments > H1segments.xml
	ligolw_segments_compat H1segments.xml
	gstlal_s5_pbh_segments_trim H1segments.xml $(TRIM) H1segmentspadded.xml

H2segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Science --result-name=datasegments > H2segments.xml
	ligolw_segments_compat H2segments.xml
	gstlal_s5_pbh_segments_trim H2segments.xml $(TRIM) H2segmentspadded.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science --result-name=datasegments > L1segments.xml
	ligolw_segments_compat L1segments.xml
	gstlal_s5_pbh_segments_trim L1segments.xml $(TRIM) L1segmentspadded.xml

vetoes.xml: H1vetoes.xml H2vetoes.xml L1vetoes.xml
	ligolw_add H1vetoes.xml H2vetoes.xml L1vetoes.xml > vetoes.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache

frame.H2.cache:
	ligo_data_find -o H -t H2_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H2.cache

frame.L1.cache:
	ligo_data_find -o L -t L1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache

frame.cache: frame.H1.cache frame.H2.cache frame.L1.cache
	cat frame.H1.cache frame.H2.cache frame.L1.cache > frame.cache

segments.xml: H1segmentspadded.xml H2segmentspadded.xml L1segmentspadded.xml
	ligolw_add --output segments.xml H1segmentspadded.xml H2segmentspadded.xml L1segmentspadded.xml
	ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segments.xml

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
	rm -r plots $(WEBDIR) *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite *.html *.css *.js Images
