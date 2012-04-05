# controls triggers
START = 871147452
STOP =  871247452
PSD_FILE = ../bank/reference_psd.xml.gz
H1_BANK_CACHE = ../bank/H1_bank.cache
H2_BANK_CACHE = ../bank/H2_bank.cache
L1_BANK_CACHE = ../bank/L1_bank.cache


# controls segments and injections
# Change seed each week
SEED = 1
MIN_DIST = 10000
MAX_DIST = 200000
MIN_MASS1 = 10
MAX_MASS1 = 100
MIN_MASS2 = 10
MAX_MASS2 = 100
MIN_TOTAL_MASS = 20
MAX_TOTAL_MASS = 200
FLOW = 39
TRIM = 16

NUMBANKS = 4

all : injections

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=0:3.14159:3.14159 tisi.xml

dag : vetoes.xml segments.xml frame.cache tisi.xml
	gstlal_s5_pbh_trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache frame.cache --frame-segments-file segments.xml  --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes vetoes.xml --time-slide-file tisi.xml --control-peak-time 4 --num-banks $(NUMBANKS) --fir-stride 4

injections : vetoes.xml segments.xml frame.cache injections.xml
	gstlal_s5_pbh_trigger_pipe --injections injections.xml --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache frame.cache --frame-segments-file segments.xml --frame-segments-name datasegments --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes vetoes.xml --control-peak-time 4 --num-banks $(NUMBANKS) --fir-stride 4

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
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science --result-name=datasegments > H1segments.xml
	ligolw_segments_compat H1segments.xml
	gstlal_segments_trim H1segments.xml --trim $(TRIM) --output H1segmentspadded.xml

H2segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Science --result-name=datasegments > H2segments.xml
	ligolw_segments_compat H2segments.xml
	gstlal_segments_trim H2segments.xml --trim $(TRIM) --output H2segmentspadded.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science --result-name=datasegments > L1segments.xml
	ligolw_segments_compat L1segments.xml
	gstlal_segments_trim L1segments.xml --trim $(TRIM) --output L1segmentspadded.xml

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

injections.xml:
	lalapps_inspinj \
		--output injections.xml \
		--seed $(SEED) \
		--f-lower 10 \
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
		--waveform  EOBNRpseudoFourPN \
		--taper-injection start \
		--disable-spin

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml
