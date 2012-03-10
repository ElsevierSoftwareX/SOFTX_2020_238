START = 871850000
STOP =  871860000
IFO = V1
DESIGNPSD = AdV_slide11_solid.txt
CHANNEL = STRAIN:16384
SAMPLE_RATE = 16384
DELTAF = 0.015625
FRAMELENGTH = 4096

all : color

$(IFO)recolor_psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument ${IFO} --output $(IFO)recolor_psd.xml.gz --invert ${DESIGNPSD} --df ${DELTAF}

color : $(IFO)recolor_psd.xml.gz
	gstlal_color_frames --gps-start-time $(START) --gps-end-time $(STOP) --instrument V1 --color-psd $(IFO)recolor_psd.xml.gz --verbose --sample-rate $(SAMPLE_RATE) --channel-name ${CHANNEL} --duration $(FRAMELENGTH)

realclean :
	rm -fr $(IFO)recolor_psd.xml.gz $(CHANNEL)

