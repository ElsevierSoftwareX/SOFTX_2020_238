START = 871147452
STOP =  871149452

all : color


H1recolorpsd.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output H1recolorpsd.xml.gz --invert ZERO_DET_high_P.txt

color : H1recolorpsd.xml.gz
	gstlal_color_frames --gps-start-time $(START) --gps-end-time $(STOP) --instrument H1 --color-psd H1recolorpsd.xml.gz --verbose

realclean :
	rm -r *.gz 
