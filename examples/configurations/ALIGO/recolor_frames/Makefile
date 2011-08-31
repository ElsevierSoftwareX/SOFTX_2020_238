MIN_SEG_LENGTH = 1800 # seconds, i.e. a half hour
START = 871147452
STOP =  871247452
SEED = 1234
LOCAL_RATE = 12.0 # mergers per Mpc^3 per Myr, corresponds to "optimistic rate"
MAX_DIST = 1000 # Mpc
FLOW = 14
TRIM = 16

all : dag

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
	gstlal_injections_by_local_rate \
	--seed $(SEED) \
	--flower $(FLOW) \
	--gps-start-time $(START) \
	--gps-end-time $(STOP) \
	--local-rate $(LOCAL_RATE) \
	--output injections.xml \
	--max-distance $(MAX_DIST)

H1recolorpsd.xml.gz : 
	./inverse_psd_xml_from_asd_txt H1 ZERO_DET_high_P.txt

H2recolorpsd.xml.gz : 
	./inverse_psd_xml_from_asd_txt H2 ZERO_DET_high_P.txt

L1recolorpsd.xml.gz : 
	./inverse_psd_xml_from_asd_txt L1 ZERO_DET_high_P.txt

recolor_psd.xml.gz: H1recolorpsd.xml.gz H2recolorpsd.xml.gz L1recolorpsd.xml.gz
	ligolw_add H1recolorpsd.xml.gz H2recolorpsd.xml.gz L1recolorpsd.xml.gz | gzip > recolor_psd.xml.gz

dag : vetoes.xml segments.xml frame.cache injections.xml recolor_psd.xml.gz
	./recolor_pipe --recolor-psd recolor_psd.xml.gz --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE)
	
realclean :
	rm -r *.sub *.dag* *.cache *.sh *reference_psd.xml.gz *segments.xml.gz logs *.xml 
