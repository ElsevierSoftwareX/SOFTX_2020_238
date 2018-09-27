
#################################
### Inputs for user to modify ###
#################################

# which interferometer (H or L)
IFO = L
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O2

START = 1183248000
END = 1183252096
SHMRUNTIME = 400
# How much time does the calibration need to settle at the start and end?
PLOT_WARMUP_TIME = 220
PLOT_COOLDOWN_TIME = 64

GDSCONFIGS = ../../config_files/O2/L1/tests/L1GDS_1222058826_frames.ini
DCSCONFIGS = ../../config_files/O2/L1/tests/L1DCS_AllCorrections_Cleaning.ini
GDSTESTCONFIGS = ../../config_files/O2/H1/tests/H1GDS_LowLatency_AllCorrections_Cleaning.ini
DCSTESTCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_Cleaning.ini
GDSSHMCONFIGS = ../../config_files/PreER13/H1/tests/H1GDS_TestLatency_AllCorrections_Cleaning.ini

all: GDS_pcal2darm_plots GDS_over_C02

################################################
### These commands should change less often ###
################################################

PLOT_START = $(shell echo $(START) + $(PLOT_WARMUP_TIME) | bc)
PLOT_END = $(shell echo $(END) - $(PLOT_COOLDOWN_TIME) | bc)

$(IFO)1_raw_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_R -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C00_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C00 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C01_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C01 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C02_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_hoft_GDS_frames.cache: $(IFO)1_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_frames.cache: $(IFO)1_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_TEST*.gwf | lalapps_path2cache > $@

# In case we want to compare one calibration to another...
$(IFO)1_hoft_GDS_TEST_frames.cache: $(IFO)1_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_TEST_frames.cache: $(IFO)1_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_GDS_SHM_frames.cache: filters framesdir
	-GST_DEBUG=3 timeout $(SHMRUNTIME) gstlal_compute_strain --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=1 --frames-per-file=1 --wings=0 --config-file $(GDSSHMCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_SHM*.gwf | lalapps_path2cache > $@

GDS_pcal2darm_plots: $(IFO)1_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_raw_frames.cache --calibrated-frame-cache $(IFO)1_hoft_GDS_frames.cache --config-file '$(GDSCONFIGS)' --pcal-channel-name CAL-PCALY_TX_PD_OUT_DQ --calibrated-channel-list='GDS-CALIB_STRAIN'

DCS_pcal2darm_plots: $(IFO)1_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_raw_frames.cache --calibrated-frame-cache $(IFO)1_hoft_DCS_frames.cache --config-file '$(DCSCONFIGS)' --pcal-channel-name CAL-PCALY_TX_PD_OUT_DQ --calibrated-channel-list='DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN'

lines_ratio_DCS: $(IFO)1_hoft_DCS_frames.cache
	python demod_ratio_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-channel-name 'DCS-CALIB_STRAIN' --numerator-channel-name 'DCS-CALIB_STRAIN_CLEAN' --frequencies '35.9,36.7,331.9,1083.7;60,120,180' --magnitude-ranges '0.0,0.1;0.0,1.0' --phase-ranges '-180.0,180.0;-180.0,180.0' --plot-titles '$(IFO)1 Calibration Line Subtraction;$(IFO)1 Power Mains Line Subtraction'

latency_test: $(IFO)1_hoft_GDS_SHM_frames.cache
	python latency_plot.py --intime-file gstlal_compute_strain_timestamps_in.txt --outtime-file gstlal_compute_strain_timestamps_out.txt --plot-filename-prefix $(IFO)1GDS_latency --plot-title '$(IFO)1 Calibration Latency vs Time'

GDS_over_C02: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_GDS_frames.cache --denominator-frame-cache $(IFO)1_C02_frames.cache --numerator-channel-name GDS-CALIB_STRAIN --denominator-channel-name DCS-CALIB_STRAIN_C02 --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --plot-title 'GDS Test / C02 Transfer Function'

DCS_over_C02: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_C02_frames.cache --numerator-channel-name DCS-CALIB_STRAIN --denominator-channel-name DCS-CALIB_STRAIN_C02 --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --plot-title 'DCS Test / C02 Transfer Function'

noise_subtraction_tf_DCS: $(IFO)1_hoft_DCS_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-channel-name DCS-CALIB_STRAIN_CLEAN --denominator-channel-name DCS-CALIB_STRAIN --magnitude-min 0.0 --magnitude-max 1.5 --phase-min -20.0 --phase-max 20.0 --plot-title 'Noise Subtraction Transfer Function'

filters:
	if [ -d Filters/$(OBSRUN)/GDSFilters ]; then \
		svn up Filters/$(OBSRUN)/GDSFilters; \
	else \
		mkdir -p Filters/$(OBSRUN); \
		cd Filters/$(OBSRUN); \
		svn co https://svn.ligo.caltech.edu/svn/aligocalibration/trunk/Runs/$(OBSRUN)/GDSFilters; \
	fi

framesdir:
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/easy_raw
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/GDS
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/DCS

clean:
	rm *.gwf *.cache *.png

