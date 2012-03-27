MIN_SEG_LENGTH = 1000 # seconds, i.e. a half hour
START = 961545543
STOP =  966308734
TRIM = 16
IFO = H1
OBSERVATORY = H
SCIENCE = DMT-SCIENCE:4
FRAME = H1_LDAS_C02_L2
DESIGNPSD = early_aLIGO.txt
CHANNEL = LDAS-STRAIN
SAMPLE_RATE = 16384
DELTAF = 0.125
FRAMELENGTH = 4096
SHIFT = -61545543

all : dag

segments.xml:
	ligolw_segment_query --segment-url=https://segdb.ligo.caltech.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=${IFO}:${SCIENCE}  --result-name=datasegments > ${IFO}segments.xml
	ligolw_segments_compat ${IFO}segments.xml
	gstlal_segments_trim --trim $(TRIM) --output ${IFO}segmentspadded.xml ${IFO}segments.xml
	ligolw_add --output segments.xml ${IFO}segmentspadded.xml

frame.cache:
	ligo_data_find -o ${OBSERVATORY} -t ${FRAME} -l  -s $(START) -e $(STOP) --url-type file > frame.cache

recolor_psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument ${IFO} --output recolor_psd.xml.gz --invert ${DESIGNPSD} --df ${DELTAF}

dag : segments.xml frame.cache recolor_psd.xml.gz
	./recolor_pipe --recolor-psd recolor_psd.xml.gz --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache ${IFO}=frame.cache --channel ${CHANNEL} --sample-rate ${SAMPLE_RATE} --df ${DELTAF} --frame-length ${FRAMELENGTH} --shift ${SHIFT}
	
realclean :
	rm -r *.sub *.dag* *.cache *.sh *reference_psd.xml.gz *segments.xml.gz logs *.xml *.xml.gz 
