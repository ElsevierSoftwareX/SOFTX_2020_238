MIN_SEG_LENGTH = 1800 # seconds, i.e. a half hour
START = 871147452
STOP =  871747452
SEED = 1
LOCAL_RATE = 12.0 # mergers per Mpc^3 per Myr, corresponds to "optimistic rate"
MAX_DIST = 500 # Mpc
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
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  --result-name=datasegments > H1segments.xml
	ligolw_segments_compat H1segments.xml
	gstlal_segments_trim --trim $(TRIM) --output H1segmentspadded.xml H1segments.xml

H2segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Science --result-name=datasegments > H2segments.xml
	ligolw_segments_compat H2segments.xml
	gstlal_segments_trim --trim $(TRIM) --output H2segmentspadded.xml H2segments.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science --result-name=datasegments > L1segments.xml
	ligolw_segments_compat L1segments.xml
	gstlal_segments_trim --trim $(TRIM) --output L1segmentspadded.xml L1segments.xml

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

# make a virgo psd even though we don't have virgo frames
H1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output H1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

L1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument L1 --output L1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

H2recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H2 --output H2recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

V1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument V1 --output V1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

recolor_psd.xml.gz: H1recolorpsd.xml.gz H2recolorpsd.xml.gz L1recolorpsd.xml.gz V1recolorpsd.xml.gz
	ligolw_add H1recolorpsd.xml.gz H2recolorpsd.xml.gz L1recolorpsd.xml.gz V1recolorpsd.xml.gz | gzip > recolor_psd.xml.gz

dag : vetoes.xml segments.xml frame.cache injections.xml recolor_psd.xml.gz
	./recolor_pipe --recolor-psd recolor_psd.xml.gz --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=frame.H1.cache,H2=frame.H2.cache,L1=frame.L1.cache
	
realclean :
	rm -r *.sub *.dag* *.cache *.sh *reference_psd.xml.gz *segments.xml.gz logs *.xml *.xml.gz 
