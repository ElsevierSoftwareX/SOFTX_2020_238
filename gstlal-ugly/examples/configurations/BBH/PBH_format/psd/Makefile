H1_FRAME_CACHE = ../segments/frame.H1.cache
V1_FRAME_CACHE = ../segments/frame.V1.cache
L1_FRAME_CACHE = ../segments/frame.L1.cache
SEGMENTS_FILE = ../segments/segments.xml

MIN_SEG_LENGTH = 3600

all : dag

dag :
	./psd_pipe --segment-file $(SEGMENTS_FILE) --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --channel LDAS-STRAIN
	
injections :
	./psd_pipe --injections $(INJECTION_FILE) --segment-file $(SEGMENTS_FILE) --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --channel LDAS-STRAIN

realclean :
	rm *.sub *.dag* *.cache *.sh
	rm -r logs
