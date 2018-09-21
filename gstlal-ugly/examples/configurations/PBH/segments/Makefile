START = 871147452
STOP =  871247452
SEED = 1234
MIN_DIST = 100
MAX_DIST = 10000
MIN_MASS = 0.2
MAX_MASS = 1.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
FLOW = 39
TRIM = 16

all : vetoes.xml segments.xml frame.cache

injections: vetoes.xml segments.xml frame.cache injections.xml

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
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  > H1segments.xml
	ligolw_segments_compat H1segments.xml
	./segments_trim H1segments.xml $(TRIM) H1segmentspadded.xml

H2segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Science > H2segments.xml
	ligolw_segments_compat H2segments.xml
	./segments_trim H2segments.xml $(TRIM) H2segmentspadded.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science > L1segments.xml
	ligolw_segments_compat L1segments.xml
	./segments_trim L1segments.xml $(TRIM) L1segmentspadded.xml

vetoes.xml: H1vetoes.xml H2vetoes.xml L1vetoes.xml
	ligolw_add --ilwdchar-compat H1vetoes.xml H2vetoes.xml L1vetoes.xml > vetoes.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache

frame.H2.cache:
	ligo_data_find -o H -t H2_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H2.cache

frame.L1.cache:
	ligo_data_find -o L -t L1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache

frame.cache: frame.H1.cache frame.H2.cache frame.L1.cache
	cat frame.H1.cache frame.H2.cache frame.L1.cache > frame.cache

segments.xml: H1segmentspadded.xml H2segmentspadded.xml L1segmentspadded.xml
	ligolw_add --ilwdchar-compat --output segments.xml H1segmentspadded.xml H2segmentspadded.xml L1segmentspadded.xml

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
		--min-mass1 $(MIN_MASS) \
		--max-mass1 $(MAX_MASS) \
		--min-mass2 $(MIN_MASS) \
		--max-mass2 $(MAX_MASS) \
		--min-mtotal $(MIN_TOTAL_MASS) \
		--max-mtotal $(MAX_TOTAL_MASS) \
		--waveform GeneratePPNtwoPN \
		--taper-injection start \
		--disable-spin

realclean:
	rm -r *.xml *.cache
