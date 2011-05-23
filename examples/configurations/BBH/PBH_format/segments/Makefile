START = 871147452
STOP = 871753660
SEED = 1234
MIN_DIST = 100
MAX_DIST = 10000
MIN_MASS = 0.2
MAX_MASS = 1.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
FLOW = 35

all : H1vetoes.xml V1vetoes.xml L1vetoes.xml segments.xml

ninja : frame.H1.cache frame.L1.cache frame.V1.cache segments.xml

injections: H1vetoes.xml V1vetoes.xml L1vetoes.xml segments.xml injections.xml

H1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Injection --result-name=vetoes > H1vetoes.xml
	ligolw_segments_compat H1vetoes.xml

V1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=V1:Injection --result-name=vetoes > V1vetoes.xml
	ligolw_segments_compat V1vetoes.xml

L1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Injection --result-name=vetoes > L1vetoes.xml
	ligolw_segments_compat L1vetoes.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_NINJA_HIGHMASS_TEST2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache
	./cache_to_segments frame.H1.cache segments.H1.xml

frame.V1.cache:
	ligo_data_find -o V -t V1_NINJA_HIGHMASS_TEST2 -l  -s $(START) -e $(STOP) --url-type file > frame.V1.cache
	./cache_to_segments frame.V1.cache segments.V1.xml

frame.L1.cache:
	ligo_data_find -o L -t L1_NINJA_HIGHMASS_TEST2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache
	./cache_to_segments frame.L1.cache segments.L1.xml

segments.xml: frame.H1.cache frame.V1.cache frame.L1.cache
	ligolw_add --output segments.xml segments.H1.xml segments.V1.xml segments.L1.xml

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
		--disable-spin

realclean:
	rm -r *.xml *.cache
