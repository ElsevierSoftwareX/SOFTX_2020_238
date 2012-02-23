START = 863557214
STOP =  875232014
FLOW = 20
L_TYPE = RDS_C03_L2
L_CHANNEL = LSC-STRAIN
V_TYPE = HrecV2_16384Hz
V_CHANNEL = h_16384Hz

all : vetoes.xml segments.xml frame.cache

H1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Injection --result-name=vetoes > H1vetoes.xml
	ligolw_segments_compat H1vetoes.xml

L1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Injection --result-name=vetoes > L1vetoes.xml
	ligolw_segments_compat L1vetoes.xml

V1vetoes.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=V1:Injection --result-name=vetoes > V1vetoes.xml
	ligolw_segments_compat V1vetoes.xml

H1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  > H1segments.xml
	ligolw_segments_compat H1segments.xml
	./segments_trim H1segments.xml $(TRIM) H1segmentspadded.xml

L1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=L1:Science > L1segments.xml
	ligolw_segments_compat L1segments.xml
	./segments_trim L1segments.xml $(TRIM) L1segmentspadded.xml

V1segmentspadded.xml:
	ligolw_segment_query --segment-url https://metaserver.phy.syr.edu -q --gps-start-time $(START) --gps-end-time $(STOP) --include-segments=V1:Science > V1segments.xml
	ligolw_segments_compat V1segments.xml
	./segments_trim V1segments.xml $(TRIM) V1segmentspadded.xml

vetoes.xml: H1vetoes.xml L1vetoes.xml V1vetoes.xml
	ligolw_add H1vetoes.xml L1vetoes.xml V1vetoes.xml > vetoes.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_$(L_TYPE) -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache

frame.L1.cache:
	ligo_data_find -o L -t L1_$(L_TYPE) -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache

frame.V1.cache:
	ligo_data_find -o V -t $(V_TYPE) -l  -s $(START) -e $(STOP) --url-type file > frame.V1.cache

frames.cache : frame.H1.cache frame.L1.cache frame.V1.cache
	cat frame.H1.cache  frame.L1.cache frame.V1.cache > frames.cache

segments.xml: H1segmentspadded.xml L1segmentspadded.xml V1segmentspadded.xml
	ligolw_add --output segments.xml H1segmentspadded.xml L1segmentspadded.xml V1segmentspadded.xml

realclean:
	rm -r *.xml *.cache
