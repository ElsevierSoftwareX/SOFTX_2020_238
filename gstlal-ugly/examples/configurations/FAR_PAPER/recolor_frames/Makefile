MIN_SEG_LENGTH = 1800 # seconds, i.e. a half hour
START = 871147452
STOP =  871747452

SEED = 2
BNS_WAVEFORM = TaylorT4threePointFivePN
NSBH_WAVEFORM = TaylorT4threePointFivePN
BNS_LOCAL_RATE = 12 # mergers per Mpc^3 per Myr, corresponds to "optimistic rate"
NSBH_LOCAL_RATE = 0.005 # mergers per Mpc^3 per Myr, corresponds to "realistic rate"
BNS_MAX_DIST = 200 # Mpc
NSBH_MAX_DIST = 5000 # Mpc
NS_MIN_MASS = 1.0
NS_MAX_MASS = 2.0
BH_MIN_MASS = 5.0
BH_MAX_MASS = 20.0
BH_MIN_SPIN = 0.0
BH_MAX_SPIN = 1.0
FLOW = 10

TRIM = 16

all : dag

H1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Injection --result-name=vetoes > H1vetoes.xml
	ligolw_segments_compat H1vetoes.xml

L1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Injection --result-name=vetoes > L1vetoes.xml
	ligolw_segments_compat L1vetoes.xml

H1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  --result-name=datasegments > H1segments.xml
	ligolw_segments_compat H1segments.xml
	gstlal_segments_trim --trim $(TRIM) --output H1segmentspadded.xml H1segments.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science --result-name=datasegments > L1segments.xml
	ligolw_segments_compat L1segments.xml
	gstlal_segments_trim --trim $(TRIM) --output L1segmentspadded.xml L1segments.xml

vetoes.xml: H1vetoes.xml L1vetoes.xml
	ligolw_add H1vetoes.xml L1vetoes.xml > vetoes.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache

frame.L1.cache:
	ligo_data_find -o L -t L1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache

frame.cache: frame.H1.cache frame.L1.cache
	cat frame.H1.cache frame.L1.cache > frame.cache
	./cache_to_segments frame.cache recolored_segments.xml.gz

segments.xml: H1segmentspadded.xml  L1segmentspadded.xml
	ligolw_add --output segments.xml H1segmentspadded.xml L1segmentspadded.xml

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
	--output $@ && gstlal_injections_by_local_rate_plot $@

H1psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output $@ ZERO_DET_high_P.txt

L1psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument L1 --output $@ ZERO_DET_high_P.txt

psd.xml.gz: H1psd.xml.gz L1psd.xml.gz
	ligolw_add H1psd.xml.gz L1psd.xml.gz | gzip > recolor_psd.xml.gz

H1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output H1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

L1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument L1 --output L1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

recolor_psd.xml.gz: H1recolorpsd.xml.gz L1recolorpsd.xml.gz
	ligolw_add H1recolorpsd.xml.gz L1recolorpsd.xml.gz | gzip > recolor_psd.xml.gz

dag : vetoes.xml segments.xml frame.cache injections.xml recolor_psd.xml.gz
	./recolor_pipe --recolor-psd recolor_psd.xml.gz --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=frame.H1.cache,L1=frame.L1.cache
	
finish : psd.xml.gz
	ls H1-LSC-STRAIN/*.gwf | lalapps_path2cache > recolor_frame.cache
	ls L1-LSC-STRAIN/*.gwf | lalapps_path2cache >> recolor_frame.cache
	./cache_to_segments recolor_frame.cache recolor_segments.xml.gz
realclean :
	rm -rf *.sub *.dag* *.cache *.sh *reference_psd.xml.gz *segments.xml.gz logs *.xml *.xml.gz  H1-LSC-STRAIN L1-LSC-STRAIN
