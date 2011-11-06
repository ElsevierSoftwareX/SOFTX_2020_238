START = 871147452
STOP =  871252452
LOCAL_RATE = 12.0 # mergers per Mpc^3 per Myr, corresponds to "optimistic rate"
MAX_DIST = 1000 # Mpc
SEED = 1
FLOW = 14


all : frames.cache

H1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument H1 --write-psd H1psd.xml.gz --verbose
	ls H1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.H1.cache

H2-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument H2 --write-psd H2psd.xml.gz --verbose
	ls H2-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.H2.cache

L1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument L1 --write-psd L1psd.xml.gz --verbose
	ls L1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.L1.cache

V1-LSC-STRAIN : injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections injections.xml --instrument V1 --write-psd V1psd.xml.gz --verbose
	ls V1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.V1.cache

frames.cache : H1-LSC-STRAIN H2-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN
	cat frame.H1.cache frame.H2.cache frame.L1.cache frame.V1.cache > frames.cache

injections.xml:
	gstlal_injections_by_local_rate \
	--seed $(SEED) \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--local-rate $(LOCAL_RATE) \
	--output injections.xml \
	--max-distance $(MAX_DIST)

realclean: 
	rm -rf injections.xml  H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz H1-LSC-STRAIN H2-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN frame.*.cache
