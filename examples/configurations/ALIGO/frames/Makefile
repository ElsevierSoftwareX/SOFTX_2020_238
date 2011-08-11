START = 871147452
STOP =  871247452
SEED = 1
MIN_DIST = 100000
MAX_DIST = 100000
MIN_MASS1 = 1.4
MAX_MASS1 = 1.4
MIN_MASS2 = 1.4
MAX_MASS2 = 1.4
MIN_TOTAL_MASS = 2.8
MAX_TOTAL_MASS = 2.8
FLOW = 14


all : frames.cache

H1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument H1 --write-psd H1psd.xml.gz --verbose
	ls H1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.H1.cache

L1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument L1 --write-psd L1psd.xml.gz --verbose
	ls L1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.L1.cache

V1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument V1 --write-psd V1psd.xml.gz --verbose
	ls V1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.V1.cache

frames.cache : H1-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN
	cat frame.H1.cache  frame.L1.cache frame.V1.cache > frames.cache

# do one injection for fun
injections.xml:
	lalapps_inspinj \
		--output injections.xml \
		--seed $(SEED) \
		--f-lower $(FLOW) \
		--gps-start-time $(START) \
		--gps-end-time $(STOP) \
		--t-distr uniform \
		--time-step 100000 \
		--time-interval 10000 \
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
		--waveform GeneratePPNtwoPN \
		--taper-injection start \
		--disable-spin

realclean: 
	rm -rf injections.xml  H1psd.xml.gz L1psd.xml.gz V1psd.xml.gz H1-LSC-STRAIN  L1-LSC-STRAIN V1-LSC-STRAIN frame.*.cache
