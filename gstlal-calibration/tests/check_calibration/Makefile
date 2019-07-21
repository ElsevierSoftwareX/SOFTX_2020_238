
#################################
### Inputs for user to modify ###
#################################

# which interferometer (H or L)
IFO = H
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O3

START = $(shell echo 1240617600 - 256 | bc)
#1238997618
END = $(shell echo 1240617600 + 512 + 256 | bc)
#1240876818
SHMRUNTIME = 36000
# How much time does the calibration need to settle at the start and end?
PLOT_WARMUP_TIME = 256
PLOT_COOLDOWN_TIME = 256

GDSCONFIGS = Filters/O3/GDSFilters/H1GDS_1239476409_test.ini
DCSCONFIGS = Filters/O3/GDSFilters/H1DCS_C01_1239472998_test.ini
DCSLINESCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_CleaningLines.ini
#../../config_files/O2/H1/tests/H1DCS_AllCorrections_Cleaning.ini
DCSFCCCONFIGS = ../../config_files/O2/H1/tests/H1DCS_FreqIndepAndFccCorrections_Cleaning.ini
GDSTESTCONFIGS = Filters/O3/GDSFilters/H1GDS_1239476409_testAllCorrections.ini
DCSTESTCONFIGS = Filters/O3/GDSFilters/L1DCS_C01_1239579018_test.ini
GDSSHMCONFIGS = Filters/ER14/GDSFilters/H1GDS_1234630818_latency_test.ini
GDSOLDCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_old.ini
GDSBETTERCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_better.ini
GDSBESTCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_best.ini

all: noise_subtraction_ASD_DCS

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
	fi
	ls Frames/$(OBSRUN)/$(IFO)1/easy_raw/$(IFO)-$(IFO)1_R-*-64.gwf | lalapps_path2cache > $@

$(IFO)1_C00_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C00 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C01_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C01 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_C02_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_clean_C02_frames.cache:
	gw_data_find -o $(IFO) -t $(IFO)1_CLEANED_HOFT_C02 -s $(START) -e $(END) -l --url-type file > $@

$(IFO)1_hoft_GDS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS-*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=256 --config-file $(DCSCONFIGS)
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

$(IFO)1_hoft_GDS_OLD_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSOLDCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_OLD*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_GDS_BETTER_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSBETTERCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_BETTER*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_GDS_BEST_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(GDSBESTCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_BEST*.gwf | lalapps_path2cache > $@

GDS_pcal2darm_plots: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache $(IFO)1_hoft_GDS_TEST_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_hoft_GDS_frames.cache,$(IFO)1_hoft_GDS_TEST_frames.cache' --config-file '$(GDSCONFIGS)' --pcal-channel-name CAL-PCALY_RX_PD_OUT_DQ --gstlal-channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN' --labels 'No corrections,All corrections' --pcal-time-advance 0.00006103515625

DCS_pcal2darm_plots: 
	#$(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C00_frames.cache
	python pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_hoft_DCS_frames.cache' --config-file '$(DCSCONFIGS)' --pcal-channel-name CAL-PCALY_RX_PD_OUT_DQ --gstlal-channel-list 'GDS-CALIB_STRAIN,DCS-CALIB_STRAIN' --labels 'C00,TEST' --pcal-time-advance 0.00006103515625

pcal_DCS_transfer_functions: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache $(IFO)1_hoft_DCS_FCC_frames.cache $(IFO)1_hoft_DCS_TEST_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-PCALY_TX_PD_OUT_DQ --denominator-correction y_arm_pcal_corr --numerator-correction arm_length --frequency-max 400 --numerator-frame-cache-list $(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_FCC_frames.cache,$(IFO)1_hoft_DCS_TEST_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN,DCS-CALIB_STRAIN_CLEAN --config-file $(DCSCONFIGS) --use-median --labels 'Only scalar corrections,Cavity pole corrections,All corrections'

lines_ratio_DCS: $(IFO)1_hoft_DCS_frames.cache
	python demod_ratio_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-channel-name 'DCS-CALIB_STRAIN' --numerator-channel-name 'DCS-CALIB_STRAIN_CLEAN' --frequencies '35.9,36.7,331.9,1083.7;60,120,180' --magnitude-ranges '0.0,0.1;0.0,1.0' --phase-ranges '-180.0,180.0;-180.0,180.0' --plot-titles '$(IFO)1 Calibration Line Subtraction;$(IFO)1 Power Mains Line Subtraction'

filters_tf_GDS: $(IFO)1_hoft_GDS_frames.cache
	python plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.9 --ratio-magnitude-max 1.1 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -10 --ratio-phase-max 10

filters_tf_DCS: $(IFO)1_hoft_DCS_frames.cache
	python plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.9 --ratio-magnitude-max 1.1 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -10 --ratio-phase-max 10

filters_tf_CALCS: $(IFO)1_easy_raw_frames.cache
	python plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.7 --ratio-magnitude-max 1.3 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -20 --ratio-phase-max 20

filters_tf_all: 
	#$(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache $(IFO)1_hoft_DCS_frames.cache
	python plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.8 --ratio-magnitude-max 1.2 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -20 --ratio-phase-max 20

latency_test: $(IFO)1_hoft_GDS_SHM_frames.cache
	python latency_plot.py --intime-file gstlal_compute_strain_timestamps_in.txt --outtime-file gstlal_compute_strain_timestamps_out.txt --plot-filename-prefix $(IFO)1GDS_latency --plot-title '$(IFO)1 Calibration Latency vs Time'

latency_tests: 
	#$(IFO)1_hoft_GDS_OLD_frames.cache $(IFO)1_hoft_GDS_BETTER_frames.cache
	python latency_plot.py --intime-file-list 'gstlal_compute_strain_timestamps_in_resample.txt,gstlal_compute_strain_timestamps_in_old.txt,gstlal_compute_strain_timestamps_in_better.txt' --outtime-file-list 'gstlal_compute_strain_timestamps_out_resample.txt,gstlal_compute_strain_timestamps_out_old.txt,gstlal_compute_strain_timestamps_out_better.txt' --plot-filename-prefix $(IFO)1GDS_latency --labels 'Including resampling latency,6s Tukey-windowed actuation filter,3.5s Kaiser-windowed actuation filter'

RMS_timeseries: $(IFO)1_C00_frames.cache $(IFO)1_X02_frames.cache
	python plot_rms_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_X02_frames.cache' --channel-list 'GDS-CALIB_STRAIN,DCS-CALIB_STRAIN_X02' --sample-rates '16,16' --labels 'C00,X02' --fmax 5.0

CALCS_GDS_ASD: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_easy_raw_frames.cache --hoft-frame-cache $(IFO)1_hoft_GDS_frames.cache

GDS_over_CALCS: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-DELTAL_EXTERNAL_DQ --denominator-name 'CALCS' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.9 --magnitude-max 1.1 --phase-min -180.0 --phase-max 180.0 --labels 'GDS / CALCS' --poles '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3' --zeros '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4' --gain 3994.5

CALCS_response: $(IFO)1_easy_raw_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-DARM_ERR_DBL_DQ --denominator-name 'DARM_ERR' --numerator-frame-cache-list $(IFO)1_easy_raw_frames.cache --numerator-channel-list CAL-DELTAL_EXTERNAL_DQ --numerator-name 'CALCS' --use-median --frequency-min 0.5 --frequency-max 8192 --magnitude-min 1e-15 --magnitude-max 1e-3 --phase-min -180.0 --phase-max 180.0 --labels 'CALCS response' --zeros '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3' --poles '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4'

GDS_over_C02: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'GDS / C02'

DCS_over_C02: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCS-CALIB_STRAIN --numerator-name 'DCS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'DCS / C02'

kappastimeseries_GDS: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	python timeserieskappas.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_hoft_GDS_frames.cache --channel-list 'GDS-CALIB_KAPPA_TST_REAL_NOGATE','GDS-CALIB_KAPPA_TST_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_PUM_REAL_NOGATE','GDS-CALIB_KAPPA_PUM_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_UIM_REAL_NOGATE','GDS-CALIB_KAPPA_UIM_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_C_NOGATE','GDS-CALIB_F_CC_NOGATE' --raw-channel-list 'CAL-CS_TDEP_KAPPA_TST_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_TST_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_PUM_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_PUM_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_UIM_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_UIM_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_C_OUTPUT','CAL-CS_TDEP_F_C_OUTPUT' --raw-frame-cache $(IFO)1_easy_raw_frames.cache

SRC_detuning_C00: $(IFO)1_C00_frames.cache
	python timeserieskappas.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_C00_frames.cache --channel-list 'GDS-CALIB_F_S_SQUARED,GDS-CALIB_SRC_Q_INVERSE'

noise_subtraction_ASD_DCS: $(IFO)1_hoft_DCS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_frames.cache' --channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN' --ASD-fmin 0.3

noise_subtraction_ASD_DCS_LINES: $(IFO)1_hoft_DCS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_frames.cache' --channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN' --ASD-fmin 0.3

noise_subtraction_ASD_GDS: $(IFO)1_hoft_GDS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_hoft_GDS_frames.cache --calcs-channel-name GDS-CALIB_STRAIN --hoft-frame-cache $(IFO)1_hoft_GDS_frames.cache --hoft-channel-name GDS-CALIB_STRAIN_CLEAN

noise_subtraction_ASD_C00: $(IFO)1_C00_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_C00_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN'

highpass_filter_ASD_GDS:
	#$(IFO)1_hoft_GDS_OLD_frames.cache $(IFO)1_hoft_GDS_BETTER_frames.cache $(IFO)1_hoft_GDS_BEST_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_hoft_GDS_OLD_frames.cache,$(IFO)1_hoft_GDS_BETTER_frames.cache,$(IFO)1_hoft_GDS_BEST_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN,GDS-CALIB_STRAIN' --ASD-fmin 0.1 --ASD-labels '6s Tukey-windowed actuation filter,3.5s Kaiser-windowed actuation filter,Additional high-pass in residual path'

noise_subtraction_ASD_DCH_DCS: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_clean_C02_frames.cache --calcs-channel-name DCH-CLEAN_STRAIN_C02 --hoft-frame-cache $(IFO)1_hoft_DCS_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

noise_subtraction_ASD_DCS_TEST: $(IFO)1_hoft_DCS_TEST_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --calcs-channel-name DCS-CALIB_STRAIN --hoft-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

noise_subtraction_tf_GDS: $(IFO)1_hoft_GDS_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_GDS_frames.cache --denominator-frame-cache $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN_CLEAN --denominator-channel-name GDS-CALIB_STRAIN --magnitude-min 0.0 --magnitude-max 1.5 --phase-min -20.0 --phase-max 20.0 --numerator-name 'CLEAN' --denominator-name 'STRAIN' --labels 'CALIB_STRAIN_CLEAN / CALIB_STRAIN' --use-median

noise_subtraction_tf_DCS: $(IFO)1_hoft_DCS_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN --denominator-channel-name DCS-CALIB_STRAIN --magnitude-min 0.0 --magnitude-max 1.5 --phase-min -20.0 --phase-max 20.0 --numerator-name 'CLEAN' --denominator-name 'STRAIN' --labels 'CALIB_STRAIN_CLEAN / CALIB_STRAIN' --use-median

noise_subtraction_tf_DCH_DCS: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_clean_C02_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN --denominator-channel-name DCH-CLEAN_STRAIN_C02 --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --numerator-name 'DCS' --denominator-name 'DCH' --labels 'DCS_CLEAN / DCH_CLEAN' --use-median

noise_subtraction_tf_DCS_DCH: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	python plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_clean_C02_frames.cache --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCH-CLEAN_STRAIN_C02 --denominator-channel-name DCS-CALIB_STRAIN_CLEAN --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --numerator-name 'DCH' --denominator-name 'DCS' --labels 'DCH_CLEAN / DCS_CLEAN' --use-median

noise_subtraction_range_plots_GDS: $(IFO)1_hoft_GDS_frames.cache
	python plot_BNS_range.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache-list '$(IFO)1_hoft_GDS_frames.cache,$(IFO)1_hoft_GDS_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN' --range-min 0.0 --range-max 140

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
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg *.svg

CLEAN:
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg *.svg Frames/$(OBSRUN)/$(IFO)1/easy_raw/*.gwf Frames/$(OBSRUN)/$(IFO)1/GDS/*.gwf Frames/$(OBSRUN)/$(IFO)1/DCS/*.gwf

