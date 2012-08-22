MIN_SEG_LENGTH = 600 # seconds, i.e. 10 minutes
START = 871147452
STOP =  871752252
TRIM = 16

all : dag

H1segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H1:Science  --result-name=datasegments > H1segments.xml
	ligolw_segments_compat H1segments.xml
	gstlal_segments_trim --trim $(TRIM) --output H1segmentspadded.xml H1segments.xml

H2segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=H2:Science --result-name=datasegments > H2segments.xml
	ligolw_segments_compat H2segments.xml
	gstlal_segments_trim --trim $(TRIM) --output H2segmentspadded.xml H2segments.xml

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache

frame.H2.cache:
	ligo_data_find -o H -t H2_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H2.cache

frame.cache: frame.H1.cache frame.H2.cache
	cat $^ > frame.cache

segments.xml: H1segmentspadded.xml H2segmentspadded.xml
	ligolw_add --output segments.xml $^

dag : segments.xml frame.cache
	./gstlal_H1H2_coherent_frames_pipe --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache frame.cache

finish :
	ls H-LHO_COHERENT*.gwf | lalapps_path2cache > H1H2_coherent_frames.cache
	ligolw_add --output H1H2_null_vetoes.xml.gz H1H2-NULL_VETOES-*.xml.gz
	sed '0,/H/s/H/datasegments/' H1H2_coherent_frames.cache > tmp.cache
	ligolw_segments --insert-from-lal-cache tmp.cache > H1H2_coherent_frames_segments.xml

realclean :
	rm -rf logs *.sub *.dag* *.cache *.sh *.xml *.xml.gz *.gwf 
