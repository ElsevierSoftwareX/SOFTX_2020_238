START = $(shell lalapps_tconvert January 18 2012)
STOP = $(shell lalapps_tconvert February 15 2012)
BNS_WAVEFORM = TaylorT4threePointFivePN
NSBH_WAVEFORM = SpinTaylorFramelessthreePointFivePN
BNS_LOCAL_RATE = 1.2 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
NSBH_LOCAL_RATE = 0.005 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
BNS_MAX_DIST = 1000 # Mpc
NSBH_MAX_DIST = 5000 # Mpc
NS_MIN_MASS = 1.0
NS_MAX_MASS = 2.0
BH_MIN_MASS = 5.0
BH_MAX_MASS = 20.0
BH_MIN_SPIN = 0.0
BH_MAX_SPIN = 1.0
SEED = 1
FLOW = 9


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
	--bns-waveform $(BNS_WAVEFORM) \
	--nsbh-waveform $(NSBH_WAVEFORM) \
	--bns-local-rate $(BNS_LOCAL_RATE) \
	--nsbh-local-rate $(NSBH_LOCAL_RATE) \
	--bns-max-distance $(BNS_MAX_DIST) \
	--nsbh-max-distance $(NSBH_MAX_DIST) \
	--ns-min-mass $(NS_MIN_MASS) \
	--ns-max-mass $(NS_MAX_MASS) \
	--bh-min-mass $(BH_MIN_MASS) \
	--bh-max-mass $(BH_MAX_MASS) \
	--bh-min-spin $(BH_MIN_SPIN) \
	--bh-max-spin $(BH_MAX_SPIN) \
	--output injections.xml && gstlal_injections_by_local_rate_plot injections.xml

realclean: 
	rm -rf injections.xml  H1psd.xml.gz H2psd.xml.gz L1psd.xml.gz V1psd.xml.gz H1-LSC-STRAIN H2-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN *.cache *.png
