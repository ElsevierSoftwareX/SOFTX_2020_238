
#################################
### Inputs for user to modify ###
#################################

# which interferometer (H or L)
IFO = H
# determines where to look for filters files (e.g., O1, O2, O3, ER10, ER13, ER14, PreER10, PreER13, PreER14)
OBSRUN = O2

START = $(shell echo 1185753618 | bc)
#1239997514 roaming line change
# 1269114318 start H1 pcal2darm for paper
# 1269021699 start of BB injection for paper (178 s)
# 1269020000 - 1269022600
# O2 H1 actuation timing plot: 1185753618 - 1185775218
END = $(shell echo 1185775218 | bc)
# 1269136430 end H1 pcal2darm for paper
SHMRUNTIME = 600
# How much time does the calibration need to settle at the start and end?
PLOT_WARMUP_TIME = 256
PLOT_COOLDOWN_TIME = 256

GDSCONFIGS = Filters/O3/GDSFilters/H1GDS_1265142330_no_response_corr_test.ini
LOWLATENCYCONFIGS = Filters/O3/GDSFilters/H1GDS_lowLatency_1239476409.ini
LOWLATENCYASYMMETRICCONFIGS = Filters/O3/GDSFilters/H1GDS_lowLatency_asymmetric_1239476409.ini
DCSCONFIGS = Filters/O3/GDSFilters/H1DCS_C01_1256655618_v2_test.ini
SCALAR_CORR_CONFIGS = Filters/O3/GDSFilters/H1DCS_test_1256655618_v2_SCALAR_CORR.ini
FCC_CORR_CONFIGS = Filters/O3/GDSFilters/H1DCS_test_1256655618_v2_FCC_CORR.ini
ALL_CORR_CONFIGS = Filters/O3/GDSFilters/H1DCS_test_1256655618_v2_ALL_CORR.ini
FCCFS_CORR_CONFIGS = Filters/O2/GDSFilters/H1DCS_FccFsCorrections_Cleaning.ini
ALL_CORR_CONFIGS = Filters/O2/GDSFilters/H1DCS_AllCorrections_Cleaning.ini
EXACTKAPPASCONFIGS = Filters/O3/GDSFilters/H1DCS_exactKappas_1239472998.ini
DCSLINESCONFIGS = ../../config_files/O2/H1/tests/H1DCS_AllCorrections_CleaningLines.ini
DCSFCCCONFIGS = ../../config_files/O2/H1/tests/H1DCS_FreqIndepAndFccCorrections_Cleaning.ini
GDSTESTCONFIGS = Filters/O3/GDSFilters/H1GDS_1239476409_testAllCorrections.ini
DCSTESTCONFIGS = Filters/O3/GDSFilters/H1DCS_C01_1252083618_AllCorrectionsTest.ini
DCSREALKAPPASCONFIGS = Filters/O3/GDSFilters/H1DCS_C01_1252083618_RealKappasTest.ini
DCSREALKAPPASSRCCONFIGS = Filters/O3/GDSFilters/H1DCS_C01_1252083618_RealKappasSRCTest.ini
GDSSHMCONFIGS = Filters/O3/GDSFilters/H1GDS_1258216456_testing.ini
GDSOLDCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_old.ini
GDSBETTERCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_better.ini
GDSBESTCONFIGS = Filters/ER14/GDSFilters/L1GDS_1235491416_best.ini

all: actuation_timing_plot
#pcal_DCS_transfer_functions
#TDCFs_pcal2darm_plots
#actuation_timing_plot
#DCS_BB_injection_plots

###############################################
### These commands should change less often ###
###############################################

PLOT_START = $(shell echo $(START) + $(PLOT_WARMUP_TIME) | bc)
PLOT_END = $(shell echo $(END) - $(PLOT_COOLDOWN_TIME) | bc)

O3_H1_BB_INJ_PATH = Measurements
O3_H1_BB_INJ_DIR = FullIFOSensingTFs
O3_H1_BB_PLOTS_PATH = Results
O3_H1_BB_PLOTS_DIR = DCS_BB_plots
O3_L1_BB_INJ_PATH = Measurements
O3_L1_BB_INJ_DIR = PCAL
O3_L1_BB_PLOTS_PATH = Measurements/PCAL
O3_L1_BB_PLOTS_DIR = DCS_BB_plots

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
		python3 frame_manipulator.py --ifo $(IFO)1 --gps-start-time $$EASY_FRAME_START --gps-end-time $$EASY_FRAME_END --frame-cache $(IFO)1_raw_frames.cache --frame-type $(IFO)1_R --output-path Frames/$(OBSRUN)/$(IFO)1/easy_raw/ ; \
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

$(IFO)1_lowLatency_GDS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(LOWLATENCYCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_LOWLATENCY-*.gwf | lalapps_path2cache > $@

$(IFO)1_lowLatency_asymmetric_GDS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/GDS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(LOWLATENCYASYMMETRICCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/GDS/$(IFO)-$(IFO)1GDS_LOWLATENCY_ASYMMETRIC-*.gwf | lalapps_path2cache > $@

$(IFO)1_hoft_DCS_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(DCSCONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS-*.gwf | lalapps_path2cache > $@

$(IFO)1_SCALAR_CORR_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(SCALAR_CORR_CONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_SCALAR_CORR-*.gwf | lalapps_path2cache > $@

$(IFO)1_FCC_CORR_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(FCC_CORR_CONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_FCC_CORR-*.gwf | lalapps_path2cache > $@

$(IFO)1_FCCFS_CORR_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(FCCFS_CORR_CONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_FCCFS_CORR-*.gwf | lalapps_path2cache > $@

$(IFO)1_ALL_CORR_frames.cache: $(IFO)1_easy_raw_frames.cache filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --gps-start-time $(START) --gps-end-time $(END) --frame-cache $(IFO)1_easy_raw_frames.cache --output-path Frames/$(OBSRUN)/$(IFO)1/DCS/ --frame-duration=64 --frames-per-file=1 --wings=0 --config-file $(ALL_CORR_CONFIGS)
	ls Frames/$(OBSRUN)/$(IFO)1/DCS/$(IFO)-$(IFO)1DCS_ALL_CORR-*.gwf | lalapps_path2cache > $@

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

increasing_latency_test: filters framesdir
	GST_DEBUG=3 gstlal_compute_strain --frame-duration=1 --frames-per-file=1 --config-file $(GDSSHMCONFIGS)

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
	python3 pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_hoft_GDS_frames.cache,$(IFO)1_hoft_GDS_TEST_frames.cache' --config-file '$(GDSCONFIGS)' --pcal-channel-name CAL-PCALY_RX_PD_OUT_DQ --gstlal-channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN' --labels 'No corrections,All corrections' --pcal-time-advance 0.00006103515625

DCS_pcal2darm_plots: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C00_frames.cache
	python3 pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_hoft_DCS_frames.cache' --config-file '$(DCSCONFIGS)' --pcal-channel-name CAL-PCALY_RX_PD_OUT_DQ --gstlal-channel-list 'GDS-CALIB_STRAIN,DCS-CALIB_STRAIN' --labels 'C00,TEST' --pcal-time-advance 0.00006103515625

TDCFs_pcal2darm_plots: $(IFO)1_easy_raw_frames.cache $(IFO)1_SCALAR_CORR_frames.cache $(IFO)1_FCC_CORR_frames.cache $(IFO)1_FCCFS_CORR_frames.cache $(IFO)1_ALL_CORR_frames.cache
	python3 pcal2darm_timeseries.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_SCALAR_CORR_frames.cache,$(IFO)1_FCC_CORR_frames.cache,$(IFO)1_FCCFS_CORR_frames.cache,$(IFO)1_ALL_CORR_frames.cache' --config-file '$(ALL_CORR_CONFIGS)' --pcal-channel-name CAL-PCALY_RX_PD_OUT_DQ --gstlal-channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN,DCS-CALIB_STRAIN,DCS-CALIB_STRAIN' --labels '{\rm Scalars},+f_{\rm cc},+f_{\rm s}+Q,+\tau_i' --latex-labels --pcal-time-advance 0.00006103515625 --show-stats

detect_actuation_timing_change: $(IFO)1_C01_frames.cache
	python3 detect_kappa_change.py --gps-start-time $(START) --gps-end-time $(END) --ifo $(IFO)1 --frame-cache $(IFO)1_C01_frames.cache --statevector-channel DCS-CALIB_STATE_VECTOR_C01 filename $(IFO)_actuation_timing_changes_$(START)-$(END).txt

actuation_timing_plot: $(IFO)1_easy_raw_frames.cache $(IFO)1_FCCFS_CORR_frames.cache $(IFO)1_ALL_CORR_frames.cache
	python3 actuation_timing_plot.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --raw-frame-cache $(IFO)1_easy_raw_frames.cache --gstlal-frame-cache-list '$(IFO)1_FCCFS_CORR_frames.cache,$(IFO)1_ALL_CORR_frames.cache' --config-file '$(ALL_CORR_CONFIGS)' --pcal-channel-name CAL-PCALY_TX_PD_OUT_DQ --gstlal-channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN' --labels '{\rm Without} \ \tau_i,{\rm With} \ \tau_i' --latex-labels --kappa-channel-list 'DCS-CALIB_KAPPA_TST_REAL,DCS-CALIB_KAPPA_TST_IMAGINARY,DCS-CALIB_KAPPA_PU_REAL,DCS-CALIB_KAPPA_PU_IMAGINARY'
	#--pcal-time-advance 0.00006103515625

pcal_GDS_transfer_functions: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-PCALY_RX_PD_OUT_DQ --denominator-correction y_arm_pcal_corr --numerator-correction arm_length --frequency-max 400 --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --config-file $(GDSCONFIGS) --use-median --labels GDS_NO_TDCFs

pcal_DCS_transfer_functions: $(IFO)1_SCALAR_CORR_frames.cache $(IFO)1_FCC_CORR_frames.cache $(IFO)1_FCCFS_CORR_frames.cache $(IFO)1_ALL_CORR_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-PCALY_RX_PD_OUT_DQ --denominator-correction y_arm_pcal_corr --numerator-correction arm_length --frequency-max 400 --numerator-frame-cache-list $(IFO)1_SCALAR_CORR_frames.cache,$(IFO)1_FCC_CORR_frames.cache,$(IFO)1_FCCFS_CORR_frames.cache,$(IFO)1_ALL_CORR_frames.cache --numerator-channel-list DCS-CALIB_STRAIN,DCS-CALIB_STRAIN,DCS-CALIB_STRAIN,DCS-CALIB_STRAIN --config-file $(ALL_CORR_CONFIGS) --use-median --labels '{\rm Scalars},+f_{\rm cc},+f_{\rm s}+Q,+\tau_i' --latex-labels

GDS_BB_injection_plots: filters framesdir BB_inj_dir BB_plots_dir
	python3 plot_BB_injections.py --ifo $(IFO) --obs-run $(OBSRUN) --check-directory BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH)/$($(OBSRUN)_$(IFO)1_BB_INJ_DIR) --plots-directory BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH)/$($(OBSRUN)_$(IFO)1_BB_PLOTS_DIR) --update-svn --cal-versions R,HOFT_C00

DCS_BB_injection_plots: filters framesdir BB_inj_dir BB_plots_dir
	python3 plot_BB_injections.py --ifo $(IFO) --obs-run $(OBSRUN) --check-directory BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH)/$($(OBSRUN)_$(IFO)1_BB_INJ_DIR) --plots-directory BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH)/$($(OBSRUN)_$(IFO)1_BB_PLOTS_DIR) --update-svn --cal-versions R,HOFT_C00,HOFT_C01

lines_ratio_DCS: $(IFO)1_hoft_DCS_frames.cache
	python3 demod_ratio_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-channel-name 'DCS-CALIB_STRAIN' --numerator-channel-name 'DCS-CALIB_STRAIN_CLEAN' --frequencies '15.6,16.4,17.1,17.6,410.3,1083.7;2001.3,2501.3' --magnitude-ranges '0.0,1.1;0.0,1.1' --phase-ranges '-180.0,180.0;-180.0,180.0' --plot-titles '$(IFO)1 Calibration Line Subtraction;$(IFO)1 Roaming Line Subtraction'

filters_tf_GDS: $(IFO)1_hoft_GDS_frames.cache
	python3 plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.95 --ratio-magnitude-max 1.05 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -3 --ratio-phase-max 3

filters_tf_DCS: $(IFO)1_ALL_CORR_frames.cache
	python3 plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.9 --ratio-magnitude-max 1.1 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -10 --ratio-phase-max 10

filters_tf_CALCS: $(IFO)1_easy_raw_frames.cache
	python3 plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.7 --ratio-magnitude-max 1.3 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -20 --ratio-phase-max 20

filters_tf_all: $(IFO)1_easy_raw_frames.cache $(IFO)1_hoft_GDS_frames.cache $(IFO)1_hoft_DCS_frames.cache
	python3 plot_filters_transfer_function.py --tf-frequency-min 0.5 --tf-frequency-max 8192 --ratio-frequency-min 10 --ratio-frequency-max 8192 --ratio-magnitude-min 0.8 --ratio-magnitude-max 1.2 --tf-phase-min -180 --tf-phase-max 180 --ratio-phase-min -20 --ratio-phase-max 20

latency_test: $(IFO)1_hoft_GDS_SHM_frames.cache
	python3 latency_plot.py --intime-file gstlal_compute_strain_timestamps_in.txt --outtime-file gstlal_compute_strain_timestamps_out.txt --plot-filename-prefix $(IFO)1GDS_latency --plot-title '$(IFO)1 Calibration Latency vs Time'

latency_tests: $(IFO)1_hoft_GDS_OLD_frames.cache $(IFO)1_hoft_GDS_BETTER_frames.cache
	python3 latency_plot.py --intime-file-list 'gstlal_compute_strain_timestamps_in_resample.txt,gstlal_compute_strain_timestamps_in_old.txt,gstlal_compute_strain_timestamps_in_better.txt' --outtime-file-list 'gstlal_compute_strain_timestamps_out_resample.txt,gstlal_compute_strain_timestamps_out_old.txt,gstlal_compute_strain_timestamps_out_better.txt' --plot-filename-prefix $(IFO)1GDS_latency --labels 'Including resampling latency,6s Tukey-windowed actuation filter,3.5s Kaiser-windowed actuation filter'

RMS_timeseries: $(IFO)1_C00_frames.cache $(IFO)1_X02_frames.cache
	python3 plot_rms_timeseries.py --ifo $(IFO)1 --gps-end-time $(PLOT_END) --gps-start-time $(PLOT_START) --frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_X02_frames.cache' --channel-list 'GDS-CALIB_STRAIN,DCS-CALIB_STRAIN_X02' --sample-rates '16,16' --labels 'C00,X02' --fmax 5.0

CALCS_GDS_ASD: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_easy_raw_frames.cache --hoft-frame-cache $(IFO)1_hoft_GDS_frames.cache

GDS_over_CALCS: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-DELTAL_EXTERNAL_DQ --denominator-name 'CALCS' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.9 --magnitude-max 1.1 --phase-min -180.0 --phase-max 180.0 --labels 'GDS / CALCS' --poles '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3' --zeros '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4' --gain 3994.5

CALCS_response: $(IFO)1_easy_raw_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_easy_raw_frames.cache --denominator-channel-name CAL-DARM_ERR_DBL_DQ --denominator-name 'DARM_ERR' --numerator-frame-cache-list $(IFO)1_easy_raw_frames.cache --numerator-channel-list CAL-DELTAL_EXTERNAL_DQ --numerator-name 'CALCS' --use-median --frequency-min 0.5 --frequency-max 8192 --magnitude-min 1e-15 --magnitude-max 1e-3 --phase-min -180.0 --phase-max 180.0 --labels 'CALCS response' --zeros '30,0,30,0,30,0,30,0,30,0,30,0,-3.009075115760242e3,3.993177550236464e3,-3.009075115760242e3,-3.993177550236464e3,-5.839434764093102e2,6.674504477214695e3,-5.839434764093102e2,-6.674504477214695e3' --poles '0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,0.3,0,1.431097327857237e2,8.198751100282409e3,1.431097327857237e2,-8.198751100282409e3,8.574723070843939e2,1.636154629741894e4,8.574723070843939e2,-1.636154629741894e4'

GDS_over_C02: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_C02_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN --numerator-name 'GDS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'GDS / C02'

DCS_over_C02: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C02_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --denominator-frame-cache $(IFO)1_C02_frames.cache --denominator-channel-name DCS-CALIB_STRAIN_C02 --denominator-name 'C02' --numerator-frame-cache-list $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCS-CALIB_STRAIN --numerator-name 'DCS' --use-median --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --labels 'DCS / C02'

kappastimeseries_GDS: $(IFO)1_hoft_GDS_frames.cache $(IFO)1_easy_raw_frames.cache
	python3 timeserieskappas.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_hoft_GDS_frames.cache --channel-list 'GDS-CALIB_KAPPA_TST_REAL_NOGATE','GDS-CALIB_KAPPA_TST_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_PUM_REAL_NOGATE','GDS-CALIB_KAPPA_PUM_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_UIM_REAL_NOGATE','GDS-CALIB_KAPPA_UIM_IMAGINARY_NOGATE','GDS-CALIB_KAPPA_C_NOGATE','GDS-CALIB_F_CC_NOGATE' --raw-channel-list 'CAL-CS_TDEP_KAPPA_TST_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_TST_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_PUM_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_PUM_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_UIM_REAL_OUTPUT','CAL-CS_TDEP_KAPPA_UIM_IMAG_OUTPUT','CAL-CS_TDEP_KAPPA_C_OUTPUT','CAL-CS_TDEP_F_C_OUTPUT' --raw-frame-cache $(IFO)1_easy_raw_frames.cache

kappastimeseries: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_C01_frames.cache
	python3 frame_manipulator.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_C01_frames.cache --output-path txt --channel-list 'DCS-CALIB_KAPPA_TST_REAL_C01,DCS-CALIB_KAPPA_TST_IMAGINARY_C01,DCS-CALIB_KAPPA_PUM_REAL_C01,DCS-CALIB_KAPPA_PUM_IMAGINARY_C01,DCS-CALIB_KAPPA_UIM_REAL_C01,DCS-CALIB_KAPPA_UIM_IMAGINARY_C01,DCS-CALIB_KAPPA_C_C01,DCS-CALIB_F_CC_C01,DCS-CALIB_F_S_SQUARED_C01,DCS-CALIB_SRC_Q_INVERSE_C01'
	python3 frame_manipulator.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_hoft_DCS_frames.cache --output-path txt --channel-list 'DCS-CALIB_KAPPA_TST_REAL,DCS-CALIB_KAPPA_TST_IMAGINARY,DCS-CALIB_KAPPA_PUM_REAL,DCS-CALIB_KAPPA_PUM_IMAGINARY,DCS-CALIB_KAPPA_UIM_REAL,DCS-CALIB_KAPPA_UIM_IMAGINARY,DCS-CALIB_KAPPA_C,DCS-CALIB_F_CC,DCS-CALIB_F_S_SQUARED,DCS-CALIB_SRC_Q_INVERSE'
	python3 plot_kappas_from_txt.py --txt-list '$(IFO)1-DCS-CALIB_KAPPA_TST_REAL_C01.txt,$(IFO)1-DCS-CALIB_KAPPA_PUM_REAL_C01.txt,$(IFO)1-DCS-CALIB_KAPPA_UIM_REAL_C01.txt;$(IFO)1-DCS-CALIB_KAPPA_TST_REAL.txt,$(IFO)1-DCS-CALIB_KAPPA_PUM_REAL.txt,$(IFO)1-DCS-CALIB_KAPPA_UIM_REAL.txt:$(IFO)1-DCS-CALIB_KAPPA_TST_IMAGINARY_C01.txt,$(IFO)1-DCS-CALIB_KAPPA_PUM_IMAGINARY_C01.txt,$(IFO)1-DCS-CALIB_KAPPA_UIM_IMAGINARY_C01.txt;$(IFO)1-DCS-CALIB_KAPPA_TST_IMAGINARY.txt,$(IFO)1-DCS-CALIB_KAPPA_PUM_IMAGINARY.txt,$(IFO)1-DCS-CALIB_KAPPA_UIM_IMAGINARY.txt' --labels 'C01;TEST' --filename actuation_TDCFs.png
	python3 plot_kappas_from_txt.py --txt-list '$(IFO)1-DCS-CALIB_KAPPA_C_C01.txt;$(IFO)1-DCS-CALIB_KAPPA_C.txt:$(IFO)1-DCS-CALIB_F_CC_C01.txt;$(IFO)1-DCS-CALIB_F_CC.txt' --labels 'C01;TEST' --filename sensing_TDCFs.png
	python3 plot_kappas_from_txt.py --txt-list '$(IFO)1-DCS-CALIB_F_S_SQUARED_C01.txt;$(IFO)1-DCS-CALIB_F_S_SQUARED.txt:$(IFO)1-DCS-CALIB_SRC_Q_INVERSE_C01.txt;$(IFO)1-DCS-CALIB_SRC_Q_INVERSE.txt' --labels 'C01;TEST' --filename SRC_TDCFs.png

SRC_detuning_C00: $(IFO)1_C00_frames.cache
	python3 timeserieskappas.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache $(IFO)1_C00_frames.cache --channel-list 'GDS-CALIB_F_S_SQUARED,GDS-CALIB_SRC_Q_INVERSE'

noise_subtraction_ASD_DCS: $(IFO)1_ALL_CORR_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_ALL_CORR_frames.cache,$(IFO)1_ALL_CORR_frames.cache' --channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN' --ASD-fmin 1 --ASD-fmax 8192 --ratio-fmin 10 --ratio-fmax 1000 --ASD-min 1e-24 --ASD-max 1e-18 --ratio-min 0

noise_subtraction_ASD_DCS_LINES: $(IFO)1_hoft_DCS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_hoft_DCS_frames.cache,$(IFO)1_hoft_DCS_frames.cache' --channel-list 'DCS-CALIB_STRAIN,DCS-CALIB_STRAIN_CLEAN' --ASD-fmin 0.3

noise_subtraction_ASD_GDS: $(IFO)1_hoft_GDS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list $(IFO)1_hoft_GDS_frames.cache,$(IFO)1_hoft_GDS_frames.cache --channel-list GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN --ASD-fmin 0.5 --ASD-labels 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN'

noise_subtraction_ASD_C00: $(IFO)1_C00_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_C00_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN'

highpass_filter_ASD_GDS: $(IFO)1_C00_frames.cache $(IFO)1_lowLatency_GDS_frames.cache $(IFO)1_lowLatency_asymmetric_GDS_frames.cache
	./ASD_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --frame-cache-list '$(IFO)1_C00_frames.cache,$(IFO)1_lowLatency_GDS_frames.cache,$(IFO)1_lowLatency_asymmetric_GDS_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN,GDS-CALIB_STRAIN' --ASD-fmin 0.5 --ASD-labels 'C00,0.95s Latency Symmetric Filters,0.95s Latency Asymmetric Filters'

noise_subtraction_ASD_DCH_DCS: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_clean_C02_frames.cache --calcs-channel-name DCH-CLEAN_STRAIN_C02 --hoft-frame-cache $(IFO)1_hoft_DCS_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

noise_subtraction_ASD_DCS_TEST: $(IFO)1_hoft_DCS_TEST_frames.cache
	./ASD_comparison_plots --ifo $(IFO)1 --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --raw-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --calcs-channel-name DCS-CALIB_STRAIN --hoft-frame-cache $(IFO)1_hoft_DCS_TEST_frames.cache --hoft-channel-name DCS-CALIB_STRAIN_CLEAN

noise_subtraction_tf_GDS: $(IFO)1_hoft_GDS_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_GDS_frames.cache --denominator-frame-cache $(IFO)1_hoft_GDS_frames.cache --numerator-channel-list GDS-CALIB_STRAIN_CLEAN --denominator-channel-name GDS-CALIB_STRAIN --magnitude-min 0.0 --magnitude-max 1.5 --phase-min -20.0 --phase-max 20.0 --numerator-name 'CLEAN' --denominator-name 'STRAIN' --labels 'CALIB_STRAIN_CLEAN / CALIB_STRAIN' --use-median

noise_subtraction_tf_DCS: $(IFO)1_hoft_DCS_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN --denominator-channel-name DCS-CALIB_STRAIN --magnitude-min 0.0 --magnitude-max 1.5 --phase-min -20.0 --phase-max 20.0 --numerator-name 'CLEAN' --denominator-name 'STRAIN' --labels 'CALIB_STRAIN_CLEAN / CALIB_STRAIN' --use-median

noise_subtraction_tf_DCH_DCS: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_hoft_DCS_frames.cache --denominator-frame-cache $(IFO)1_clean_C02_frames.cache --numerator-channel-list DCS-CALIB_STRAIN_CLEAN --denominator-channel-name DCH-CLEAN_STRAIN_C02 --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --numerator-name 'DCS' --denominator-name 'DCH' --labels 'DCS_CLEAN / DCH_CLEAN' --use-median

noise_subtraction_tf_DCS_DCH: $(IFO)1_hoft_DCS_frames.cache $(IFO)1_clean_C02_frames.cache
	python3 plot_transfer_function.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --numerator-frame-cache $(IFO)1_clean_C02_frames.cache --denominator-frame-cache $(IFO)1_hoft_DCS_frames.cache --numerator-channel-list DCH-CLEAN_STRAIN_C02 --denominator-channel-name DCS-CALIB_STRAIN_CLEAN --magnitude-min 0.7 --magnitude-max 1.3 --phase-min -20.0 --phase-max 20.0 --numerator-name 'DCH' --denominator-name 'DCS' --labels 'DCH_CLEAN / DCS_CLEAN' --use-median

noise_subtraction_range_plots_GDS: $(IFO)1_hoft_GDS_frames.cache
	python3 plot_BNS_range.py --gps-start-time $(PLOT_START) --gps-end-time $(PLOT_END) --ifo $(IFO)1 --frame-cache-list '$(IFO)1_hoft_GDS_frames.cache,$(IFO)1_hoft_GDS_frames.cache' --channel-list 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN_CLEAN' --range-min 0.0 --range-max 140

filters:
	if [ -d Filters/$(OBSRUN)/GDSFilters ]; then \
		svn up Filters/$(OBSRUN)/GDSFilters; \
	else \
		mkdir -p Filters/$(OBSRUN); \
		cd Filters/$(OBSRUN); \
		svn co https://svn.ligo.caltech.edu/svn/aligocalibration/trunk/Runs/$(OBSRUN)/GDSFilters; \
	fi

BB_inj_dir:
	if [ -d BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH)/$($(OBSRUN)_$(IFO)1_BB_INJ_DIR) ]; then \
		svn up BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH)/$($(OBSRUN)_$(IFO)1_BB_INJ_DIR); \
	else \
		mkdir -p BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH); \
		cd BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH); \
		svn co https://svn.ligo.caltech.edu/svn/aligocalibration/trunk/Runs/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_INJ_PATH)/$($(OBSRUN)_$(IFO)1_BB_INJ_DIR); \
	fi

BB_plots_dir:
	if [ -d BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH)/$($(OBSRUN)_$(IFO)1_BB_PLOTS_DIR) ]; then \
		svn up BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH)/$($(OBSRUN)_$(IFO)1_BB_PLOTS_DIR); \
	else \
		mkdir -p BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH); \
		cd BB_injections/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH); \
		svn co https://svn.ligo.caltech.edu/svn/aligocalibration/trunk/Runs/$(OBSRUN)/$(IFO)1/$($(OBSRUN)_$(IFO)1_BB_PLOTS_PATH)/$($(OBSRUN)_$(IFO)1_BB_PLOTS_DIR); \
	fi

framesdir:
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/easy_raw
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/GDS
	mkdir -p Frames/$(OBSRUN)/$(IFO)1/DCS

clean:
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg *.svg

CLEAN:
	rm *.gwf *.cache *.png *.txt *.pdf *.jpg *.svg Frames/$(OBSRUN)/$(IFO)1/easy_raw/*.gwf Frames/$(OBSRUN)/$(IFO)1/GDS/*.gwf Frames/$(OBSRUN)/$(IFO)1/DCS/*.gwf

