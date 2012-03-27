START = 900000000
STOP =  904763191
IFO = H1
DESIGNPSD = early_aLIGO.txt
CHANNEL = GAUSSIAN
SAMPLE_RATE = 16384
DELTAF = 0.125
FRAMELENGTH = 4096


all : color


$(IFO)recolor_psd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument ${IFO} --output $(IFO)recolor_psd.xml.gz --invert ${DESIGNPSD} --df ${DELTAF}

color : $(IFO)recolor_psd.xml.gz
	gstlal_color_frames --gps-start-time $(START) --gps-end-time $(STOP) --instrument ${IFO} --color-psd $(IFO)recolor_psd.xml.gz --verbose --sample-rate $(SAMPLE_RATE) --channel-name ${CHANNEL} --duration $(FRAMELENGTH)

realclean :
	rm -fr $(IFO)recolor_psd.xml.gz $(CHANNEL)

