START = 900000000
TESTSTART = 900000032
STOP =  900005000
TESTSTOP =  900004068
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

test :
	ls H1-GAUSSIAN/*.gwf | lalapps_path2cache > frame.cache
	gstlal_reference_psd --frame-cache frame.cache --gps-start-time $(TESTSTART) --gps-end-time $(TESTSTOP) --instrument ${IFO} --write-psd test_psd.xml.gz --channel-name ${CHANNEL} --verbose
	gstlal_plot_psd test_psd.xml.gz test
	gstlal_psd_xml_from_asd_txt --instrument ${IFO} --output $(IFO)original_psd.xml.gz --df $(DELTAF) $(DESIGNPSD)
	gstlal_plot_psd $(IFO)original_psd.xml.gz original

realclean :
	rm -fr $(IFO)recolor_psd.xml.gz $(CHANNEL) H1-GAUSSIAN

