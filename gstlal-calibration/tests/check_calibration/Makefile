
#################################
### Inputs for user to modify ###
#################################

# which interferometer (H or L)
IFO = H
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O2

START = $(shell echo 1185767424 - 715 | bc)
#1225967424
#$(shell echo 1185763328 - 700 | bc)
END = $(shell echo 1185771520 + 715 | bc)
#1225968448
#$(shell echo 1185767424 + 700 | bc)
# 1185771520
SHMRUNTIME = 400
# How much time does the calibration need to settle at the start and end?
PLOT_WARMUP_TIME = 715
PLOT_COOLDOWN_TIME = 715

GDSCONFIGS = ../../config_files/PreER13/H1/tests/H1GDS_1227048227.ini
DCSCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_Cleaning.ini
DCSFCCCONFIGS = ../../config_files/O2/H1/tests/H1DCS_FreqIndepAndFccCorrections_Cleaning.ini
GDSTESTCONFIGS = ../../config_files/PreER13/H1/H1GDS_TEST_1225558818.ini
DCSTESTCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_Cleaning_TEST.ini
GDSSHMCONFIGS = ../../config_files/PreER13/H1/tests/H1GDS_1222058826_shm2frames.ini

all: noise_subtraction_ASD_DCS_TEST

###############################################
### These commands should change less often ###
###############################################

PLOT_START = $(shell echo $(START) + $(PLOT_WARMUP_TIME) | bc)
PLOT_END = $(shell echo $(END) - $(PLOT_COOLDOWN_TIME) | bc)

$(IFO)1_raw_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_R -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_easy_raw_frames.cache: $(IFO)1_raw_frames.cache framesdir
	EASY_FRAME_START=`echo "$(START) - $(START) % 64" | bc`; \
	while [[ -a Frames/$(OBSRUN)/$(IFO)1/easy_raw/$(IFO)-$(IFO)1_R-$$EASY_FRAME_START-64.gwf ]] ; do \
		EASY_FRAME_START=`echo "$$EASY_FRAME_START + 64" | bc`; \
	done; \
	EASY_FRAME_END=`echo "(($(END) - 1)) - (($(END) - 1)) % 64" | bc`; \
	while [[ -a Frames/$(OBSRUN)/$(IFO)1/easy_raw/$(IFO)-$(IFO)1_R-$$EASY_FRAME_END-64.gwf ]] ; do \
		EASY_FRAME_END=`echo "$$EASY_FRAME_END - 64" | bc`; \
	done; \
	EASY_FRAME_END=`echo "$$EASY_FRAME_END + 64" | bc`; \
	if [ $$EASY_FRAME_END -gt $$EASY_FRAME_START ] ; then \
		FrChannels $(shell sed -nr '0,/localhost/ s/.*localhost([^"]+).*/\1/p' $(IFO)1_raw_frames.cache) > frchannels.txt ; \
		python frame_manipulator.py --ifo $(IFO)1 --gps-start-time $$EASY_FRAME_START --gps-end-time $$EASY_FRAME_END --frame-cache $(IFO)1_raw_frames.cache --frame-type $(IFO)1_R --output-path Frames/$(OBSRUN)/$(IFO)1/easy_raw/ ; \
		rm frchannels.txt; \
	fi
	ls Frames/$(OBSRUN)/$(IFO)1/easy_raw/$(IFO)-$(IFO)1_R-*-64.gwf | lalapps_path2cache > $@

$(IFO)1_C00_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C00 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C01_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C01 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C02_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_hoft_GDS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS-*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS-*.gwf | lalapps_path2cache > $@

# In case we want to compare one calibration to another...
$(IFO)1_hoft_GDS_TEST_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_TEST-*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_TEST_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSTESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_TEST-*.gwf | lalapps_path2cache > $@

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

CALCS_GDS_ASD: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_easy_raw_frames.cache --hoft-frame-cache $(IFO)1_hoft_GDS_frames.cache

GDS_over_CALCS: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-DELTAL_EXTERNAL_DQ --denominator-name 'CALCS' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -180.0 --phase-max 180.0 --labels 'GDS / CALCS' --poles '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3' --zeros '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4' --gain 3994.5

GDS_over_C02: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'GDS / C02'

DCS_over_C02: $(IFO)1_hoft_DCS_FCC_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_DCS_FCC_frames.cache --numerator-channel-list DCS-CALIB_STRAIN --numerator-name 'DCS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'DCS / C02'

noise_subtraction_ASD_DCS: $(IFO)1_hoft_DCS_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_hoft_DCS_frames.cache --calcs-channel-name DCS-CALIB_STRAIN --hoft-frame-cache $(IFO)1_hoft_DCS_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

noise_subtraction_ASD_DCS_TEST: $(IFO)1_hoft_DCS_TEST_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --calcs-channel-name DCS-CALIB_STRAIN --hoft-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

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

