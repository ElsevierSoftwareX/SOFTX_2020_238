
#################################
### Inputs for user to modify ###
#################################

# which interferometer
IFO = H1
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O2

START = 1186159035
END = 1186159557
GDSCONFIGS = ../../config_files/H1GDS_LowLatency_AllCorrections_Cleaning.ini
DCSCONFIGS = ../../config_files/H1DCS_AllCorrections_Cleaning.ini

all: DCS_pcal2darm_plots

################################################
### These commands should changes less often ###
################################################

$(IFO)_raw_frames.cache:
	gw_data_find -o H -t H1_R -s $(START) -e $(END) -l --url-type file > $@

$(IFO)_C00_frames.cache:
	gw_data_find -o H -t H1_HOFT_C00 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)_C01_frames.cache:
	gw_data_find -o H -t H1_HOFT_C01 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)_C02_frames.cache:
	gw_data_find -o H -t H1_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)_hoft_GDS_frames.cache: $(IFO)_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSCONFIGS)
	ls H-H1GDS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)_hoft_DCS_frames.cache: $(IFO)_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSCONFIGS)
	ls H-H1DCS_TEST*.gwf | lalapps_path2cache > $@

GDS_pcal2darm_plots: $(IFO)_raw_frames.cache $(IFO)_hoft_GDS_frames.cache
	python pcal2darm_timeseries.py --ifo=$(IFO) --raw-frame-cache $(IFO)_raw_frames.cache --calibrated-frame-cache $(IFO)_hoft_GDS_frames.cache --config-file $(GDSCONFIGS) --calibrated-channel-list='GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN'

DCS_pcal2darm_plots: $(IFO)_raw_frames.cache $(IFO)_hoft_DCS_frames.cache
	python pcal2darm_timeseries.py --ifo=$(IFO) --raw-frame-cache $(IFO)_raw_frames.cache --calibrated-frame-cache $(IFO)_hoft_DCS_frames.cache --config-file $(DCSCONFIGS) --calibrated-channel-list='DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN'

filters:
	if [ -d Filters/$(OBSRUN)/GDSFilters ]; then \
		svn up Filters/$(OBSRUN)/GDSFilters; \
	else \
		mkdir -p Filters/$(OBSRUN); \
		cd Filters/$(OBSRUN); \
		svn co https://svn.ligo.caltech.edu/svn/aligocalibration/trunk/Runs/$(OBSRUN)/GDSFilters; \
	fi

framesdir:
	mkdir -p Frames/$(OBSRUN)/$(IFO)/easy_raw
	mkdir -p Frames/$(OBSRUN)/$(IFO)/GDS
	mkdir -p Frames/$(OBSRUN)/$(IFO)/DCS

clean:
	rm *.gwf *.cache *.png

