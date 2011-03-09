START = 871147452
STOP =  871247452
MIN_SEG_LENGTH = 3600
SEED = 1234
MIN_DIST = 1
MAX_DIST = 10000
MIN_MASS = 0.2
MAX_MASS = 1.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
FLOW = 35

all : dag

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache
	./cache_to_segments frame.H1.cache segments.H1.xml

frame.H2.cache:
	ligo_data_find -o H -t H2_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H2.cache
	./cache_to_segments frame.H2.cache segments.H2.xml

frame.L1.cache:
	ligo_data_find -o L -t L1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache
	./cache_to_segments frame.L1.cache segments.L1.xml

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

segments.xml:
	ligolw_add --output segments.xml segments.H1.xml segments.H2.xml segments.L1.xml

dag : frame.H1.cache frame.H2.cache frame.L1.cache segments.xml injections.xml
	./psd_pipe --injections injections.xml --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=frame.H1.cache,H2=frame.H2.cache,L1=frame.L1.cache
	
realclean :
	rm *.sub *.dag* *.cache *.sh segments.*xml
	rm -r logs
