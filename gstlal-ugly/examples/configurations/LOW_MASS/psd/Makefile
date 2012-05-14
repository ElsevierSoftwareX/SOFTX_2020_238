# Misc useful definitions
empty:=
space:= $(empty) $(empty)
comma:= ,

# controls triggers
IFOS = H1 L1
START = 869400000
STOP =  870600000

# Segments file names
segments_suffix := _segmentspadded.xml
SEGMENTS_FILES  := $(addsuffix $(segments_suffix),$(IFOS))

# reference psd file names
psd_suffix := _psd.xml.gz
PSD_FILES  := $(addsuffix $(psd_suffix),$(IFOS))

TRIM = 16


# Frame cache file names
frame_suffix      := _frame.cache
FRAME_CACHE_FILES := $(addsuffix $(frame_suffix),$(IFOS))

all : reference_psd.xml.gz

reference_psd.xml.gz : $(PSD_FILES)
	ligolw_add $(PSD_FILES) --output reference_psd.xml.gz

%_psd.xml.gz : frame.cache segments.xml
	gstlal_reference_psd --frame-cache frame.cache --frame-segments-file segments.xml --gps-start-time $(START) --gps-end-time $(STOP) --instrument $* --write-psd $@ --verbose
	

%_segmentspadded.xml:
	ligolw_segment_query --segment-url=https://metaserver.phy.syr.edu -q --gps-start-time ${START} --gps-end-time ${STOP} --include-segments=$*:Science --result-name=datasegments > $*segments.xml
	ligolw_segments_compat $*segments.xml
	gstlal_segments_trim $*segments.xml --trim $(TRIM) --output $@

%_frame.cache:
	#FIXME horrible hack to get the observatory, not guaranteed to work
	$(eval OBS:=$*)
	$(eval OBS:=$(subst 1,$(empty),$(OBS)))
	$(eval OBS:=$(subst 2,$(empty),$(OBS)))
	ligo_data_find -o $(OBS) -t $*_RDS_C03_L2 -l  -s $(START) -e $(STOP) --url-type file > $@

frame.cache: $(FRAME_CACHE_FILES)
	cat $(FRAME_CACHE_FILES) > frame.cache

segments.xml: $(SEGMENTS_FILES)
	ligolw_add --output segments.xml $(SEGMENTS_FILES)
	ligolw_cut --delete-column segment:segment_def_cdb --delete-column segment:creator_db --delete-column segment_definer:insertion_time segments.xml



realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite plots $(WEBDIR) *.html Images *.css *.js *.png
