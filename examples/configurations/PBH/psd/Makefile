H1_FRAME_CACHE = ../segments/frame.H1.cache
H2_FRAME_CACHE = ../segments/frame.H2.cache
L1_FRAME_CACHE = ../segments/frame.L1.cache
H1_VETOES = ../segments/H1vetoes.xml
H2_VETOES = ../segments/H2vetoes.xml
L1_VETOES = ../segments/L1vetoes.xml
SEGMENTS_FILE = ../segments/segments.xml
INJECTION_FILE = ../segments/injections.xml

MIN_SEG_LENGTH = 3600

all : dag

dag :
	./psd_pipe --segment-file $(SEGMENTS_FILE) --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),H2=$(H2_FRAME_CACHE),L1=$(L1_FRAME_CACHE)
	
injections :
	./psd_pipe --injections $(INJECTION_FILE) --segment-file $(SEGMENTS_FILE) --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),H2=$(H2_FRAME_CACHE),L1=$(L1_FRAME_CACHE)

realclean :
	rm *.sub *.dag* *.cache
	rm -r logs
