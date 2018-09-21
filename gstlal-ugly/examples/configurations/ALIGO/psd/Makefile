H1_FRAME_CACHE = ../frames/frame.H1.cache
V1_FRAME_CACHE = ../frames/frame.V1.cache
L1_FRAME_CACHE = ../frames/frame.L1.cache

MIN_SEG_LENGTH = 1

all : dag

segments.xml:
	./cache_to_segments $(H1_FRAME_CACHE) H1segments.xml.gz
	./cache_to_segments $(L1_FRAME_CACHE) L1segments.xml.gz
	./cache_to_segments $(V1_FRAME_CACHE) V1segments.xml.gz
	ligolw_add --ilwdchar-compat H1segments.xml.gz L1segments.xml.gz V1segments.xml.gz --output segments.xml

dag : segments.xml
	./psd_pipe --segment-file segments.xml --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE)
	
realclean :
	rm -r *.sub *.dag* *.cache *.sh *reference_psd.xml.gz *segments.xml.gz logs segments.xml
