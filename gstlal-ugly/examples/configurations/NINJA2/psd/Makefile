H1_FRAME_CACHE = ../frames/frame.H1.cache
V1_FRAME_CACHE = ../frames/frame.V1.cache
L1_FRAME_CACHE = ../frames/frame.L1.cache

MIN_SEG_LENGTH = 1

all : V1_psd H1_psd L1_psd V1segments.xml.gz H1segments.xml.gz L1segments.xml.gz segments.xml.gz

V1segments.xml.gz:
	./cache_to_segments $(V1_FRAME_CACHE) V1segments.xml.gz multiSeg

L1segments.xml.gz:
	./cache_to_segments $(L1_FRAME_CACHE) L1segments.xml.gz multiSeg

H1segments.xml.gz:
	./cache_to_segments $(H1_FRAME_CACHE) H1segments.xml.gz multiSeg

segments.xml.gz:
	ligolw_add --ilwdchar-compat H1segments.xml.gz L1segments.xml.gz V1segments.xml.gz --output segments.xml.gz

V1_psd : V1segments.xml.gz
	./psd_pipe --segment-file $^ --min-segment-length $(MIN_SEG_LENGTH) --frame-cache V1=$(V1_FRAME_CACHE) --channel h_16384Hz

H1_psd : H1segments.xml.gz
	./psd_pipe --segment-file $^ --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE) --channel LDAS-STRAIN

L1_psd : L1segments.xml.gz
	./psd_pipe --segment-file $^ --min-segment-length $(MIN_SEG_LENGTH) --frame-cache L1=$(L1_FRAME_CACHE) --channel LDAS-STRAIN

ligolw_add_commands: H1_psd L1_psd V1_psd
	ligolw_add --ilwdchar-compat -v --output H1_reference_psd.xml.gz H1*.xml.gz
	ligolw_add --ilwdchar-compat -v --output L1_reference_psd.xml.gz L1*.xml.gz
	ligolw_add --ilwdchar-compat -v --output V1_reference_psd.xml.gz V1*.xml.gz
	ligolw_add --ilwdchar-compat -v --output reference_psd.xml.gz *1_reference_psd.xml.gz

#injections :
#	./psd_pipe --injections $(INJECTION_FILE) --segment-file $(SEGMENTS_FILE) --min-segment-length $(MIN_SEG_LENGTH) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --channel LDAS-STRAIN

realclean :
	rm *.sub *.dag* *.cache *.sh *.xml *.xml.gz *.out
	rm -r logs
