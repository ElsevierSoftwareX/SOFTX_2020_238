START = 871147452
STOP =  871247452
MIN_SEG_LENGTH = 3600
# IFOS = H

# Template bank parameters
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 1023.0
SAMPLE_RATE = 2048

all : dag

frame.H1.cache:
	ligo_data_find -o H -t H1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H1.cache
	./cache_to_segments frame.H1.cache segments.H1.xml

frame.H2.cache:
	ligo_data_find -o H -t H2_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.H2.cache
	./cache_to_segments frame.H2.cache segments.H2.xml

frame.L1.cache:
	ligo_data_find -o L -t L1_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > frame.L1.cache
	./cache_to_segments frame.L1.cache segments.L1.xml

segments.xml:
	ligolw_add --output segments.xml segments.H1.xml segments.H2.xml segments.L1.xml

dag : frame.H1.cache frame.H2.cache frame.L1.cache segments.xml
	./psd_pipe --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=frame.H1.cache,H2=frame.H2.cache,L1=frame.L1.cache
	
realclean :
	rm *.sub *.dag* *.cache *.sh segments.*xml
	rm -r logs
