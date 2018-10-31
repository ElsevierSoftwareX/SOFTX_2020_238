
#################################
### Inputs for user to modify ###
#################################

# which interferometer (H or L)
IFO = H
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O2

START = 1186944000
END = 1186945024
SHMRUNTIME = 400
# How much time does the calibration need to settle at the start and end?
PLOT_WARMUP_TIME = 300
PLOT_COOLDOWN_TIME = 64

GDSCONFIGS = ../../config_files/PreER13/L1/tests/L1GDS_1222058826_frames.ini
DCSCONFIGS = ../../config_files/O2/H1/tests/H1DCS_FreqIndepCorrections_Cleaning.ini
DCSFCCCONFIGS = ../../config_files/O2/H1/tests/H1DCS_FreqIndepAndFccCorrections_Cleaning.ini
GDSTESTCONFIGS = ../../config_files/O2/H1/tests/H1GDS_LowLatency_AllCorrections_Cleaning.ini
DCSTESTCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_Cleaning.ini
GDSSHMCONFIGS = ../../config_files/PreER13/H1/tests/H1GDS_1222058826_shm2frames.ini

all: DCS_over_C02

###############################################
### These commands should change less often ###
###############################################

PLOT_START = $(shell echo $(START) + $(PLOT_WARMUP_TIME) | bc)
PLOT_END = $(shell echo $(END) - $(PLOT_COOLDOWN_TIME) | bc)

$(IFO)1_raw_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_R -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_easy_raw_frames.cache: $(IFO)1_raw_frames.cache framesdir
	FrChannels $(shell sed -nr '0,/localhost/ s/.*localhost([^"]+).*/\1/p' $(IFO)1_raw_frames.cache) > frchannels.txt
	python frame_manipulator.py --ifo $(IFO)1 --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_raw_frames.cache --frame-type $(IFO)1_R --output-path Frames/$(OBSRUN)/$(IFO)1/easy_raw/
	ls Frames/$(OBSRUN)/$(IFO)1/easy_raw/$(IFO)-$(IFO)1_R*.gwf | lalapps_path2cache > $@
	rm frchannels.txt

$(IFO)1_C00_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C00 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C01_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C01 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C02_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_hoft_GDS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS*.gwf | lalapps_path2cache > $@

# In case we want to compare one calibration to another...
$(IFO)1_hoft_GDS_TEST_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_TEST_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_TEST*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_FCC_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSFCCCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_FCC*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_GDS_SHM_frames.cache: filters framesdir
	-GST_DEBUG=3 timeout $(SHMRUNTIME) gstlal_compute_strain --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=1 --frames-per-file=1 --wings=0 --config-file $(GDSSHMCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_SHM*.gwf | lalapps_path2cache > $@

GDS_pcal2darm_plots: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --config-file '$(GDSCONFIGS)' --pcal-channel-name CAL-PCALY_TX_PD_OUT_DQ --gstlal-channel-list GDS-CALIB_STRAIN --labels GDS-CALIB_STRAIN

DCS_pcal2darm_plots: (IFO)1_easy_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache $(IFO)1_hoft_DCS_FCC_frames.cache $(IFO)1_hoft_DCS_TEST_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list $(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_FCC_frames.cache,$(IFO)1_hoft_DCS_TEST_frames.cache --config-file '$(DCSCONFIGS)' --pcal-channel-name CAL-PCALY_TX_PD_OUT_DQ --gstlal-channel-list DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN --labels 'Scalars,Cavity pole,All'

pcal_DCS_transfer_functions: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache $(IFO)1_hoft_DCS_FCC_frames.cache $(IFO)1_hoft_DCS_TEST_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-PCALY_TX_PD_OUT_DQ --denominator-correction y_arm_pcal_corr --numerator-correction arm_length --frequency-max 400 --numerator-frame-cache-list $(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_FCC_frames.cache,$(IFO)1_hoft_DCS_TEST_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN --config-file $(DCSCONFIGS) --use-median --labels 'Only scalar corrections,Cavity pole corrections,All corrections'

lines_ratio_DCS: $(IFO)1_hoft_DCS_frames.cache
	python demod_ratio_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-channel-name 'DCS-CALIB_STRAIN' --numerator-channel-name 'DCS-CALIB_STRAIN_CLEAN' --frequencies '35.9,36.7,331.9,1083.7;60,120,180' --magnitude-ranges '0.0,0.1;0.0,1.0' --phase-ranges '-180.0,180.0;-180.0,180.0' --plot-titles '$(IFO)1 Calibration Line Subtraction;$(IFO)1 Power Mains Line Subtraction'

latency_test: $(IFO)1_hoft_GDS_SHM_frames.cache
	python latency_plot.py --intime-file gstlal_compute_strain_timestamps_in.txt --outtime-file gstlal_compute_strain_timestamps_out.txt --plot-filename-prefix $(IFO)1GDS_latency --plot-title '$(IFO)1 Calibration Latency vs Time'

GDS_over_C02: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'GDS / C02'

DCS_over_C02: $(IFO)1_hoft_DCS_FCC_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --numerator-frame-cache-list $(IFO)1_hoft_DCS_FCC_frames.cache --numerator-channel-list DCS-CALIB_STRAIN --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'DCS / C02'

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
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg

CLEAN:
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg Frames/$(OBSRUN)/$(IFO)1/easy_raw/*.gwf Frames/$(OBSRUN)/$(IFO)1/GDS/*.gwf Frames/$(OBSRUN)/$(IFO)1/DCS/*.gwf

