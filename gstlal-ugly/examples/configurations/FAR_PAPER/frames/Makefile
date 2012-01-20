START = 871147452
STOP =  871247452
LOCAL_RATE = 12.0 # mergers per Mpc^3 per Myr, corresponds to "optimistic rate"
MAX_DIST = 1000 # Mpc
SEED = 1
FLOW = 14


all : frames.cache

H1-LSC-STRAIN : H1injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections H1injections.xml --instrument H1 --write-psd H1psd.xml.gz --verbose
	ls H1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.H1.cache

L1-LSC-STRAIN : L1injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections L1injections.xml --instrument L1 --write-psd L1psd.xml.gz --verbose
	ls L1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.L1.cache

V1-LSC-STRAIN : V1injections.xml
	gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(STOP) --injections V1injections.xml --instrument V1 --write-psd V1psd.xml.gz --verbose
	ls V1-LSC-STRAIN/*.gwf | lalapps_path2cache > frame.V1.cache

frames.cache : H1-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN
	cat frame.H1.cache  frame.L1.cache frame.V1.cache > frames.cache

#FIXME needs to be updated to new command line arguments
injections.xml:
	gstlal_injections_by_local_rate \
	--seed $(SEED) \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--local-rate $(LOCAL_RATE) \
	--output injections.xml \
	--max-distance $(MAX_DIST)

#FIXME tweak the amplitudes in this program???
H1injections.xml: injections.xml
	./gstlal_glitch_population --gps-start-time $(START) --gps-end-time $(STOP) H1glitches.xml && \
	ligolw_add -o H1injections.xml H1glitches.xml injections.xml

L1injections.xml: injections.xml
	./gstlal_glitch_population --gps-start-time $(START) --gps-end-time $(STOP) L1glitches.xml && \
	ligolw_add -o L1injections.xml L1glitches.xml injections.xml

V1injections.xml: injections.xml
	./gstlal_glitch_population --gps-start-time $(START) --gps-end-time $(STOP) V1glitches.xml && \
	ligolw_add -o V1injections.xml V1glitches.xml injections.xml

realclean: 
	rm -rf *glitches.xml *injections.xml  H1psd.xml.gz L1psd.xml.gz V1psd.xml.gz H1-LSC-STRAIN  L1-LSC-STRAIN V1-LSC-STRAIN frame.*.cache
