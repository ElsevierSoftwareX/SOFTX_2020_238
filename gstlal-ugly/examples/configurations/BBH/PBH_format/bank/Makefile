START = 871147452
STOP = 871753660
IFOS = V

# Template bank parameters
MIN_MASS = 1.0
MAX_MASS = 99.0
MIN_TOTAL_MASS = 25.0
MAX_TOTAL_MASS = 100.0
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 1023.0
SAMPLE_RATE = 2048
NUM_SPLIT_TEMPLATES = 500
MM = 0.97

V1-TMPLTBANK-871154847-2048.xml: ../segments/frame.V1.cache 
        lalapps_tmpltbank \
                --disable-compute-moments \
                --grid-spacing Hexagonal \
                --dynamic-range-exponent 69.0 \
                --enable-high-pass $(HIGH_PASS_FREQ) \
                --high-pass-order 8 \
                --strain-high-pass-order 8 \
                --minimum-mass $(MIN_MASS) \
                --maximum-mass $(MAX_MASS) \
                --min-total-mass $(MIN_TOTAL_MASS) \
                --max-total-mass $(MAX_TOTAL_MASS) \
                --gps-start-time 871154847 \
                --gps-end-time 871156895 \
                --standard-candle \
                --calibrated-data real_4 \
                --candle-mass1 1 \
                --candle-mass2 1 \
                --channel-name V1:LDAS-STRAIN \
                --space Tau0Tau3 \
                --number-of-segments 15 \
                --minimal-match $(MM) \
                --candle-snr 8 \
                --debug-level 33 \
                --high-pass-attenuation 0.1 \
                --min-high-freq-cutoff LRD \
                --segment-length 524288 \
                --low-frequency-cutoff $(LOW_FREQUENCY_CUTOFF) \
                --pad-data 8 \
                --num-freq-cutoffs 1 \
                --sample-rate $(SAMPLE_RATE) \
                --high-frequency-cutoff $(HIGH_FREQUENCY_CUTOFF) \
                --resample-filter ldas \
                --strain-high-pass-atten 0.1 \
                --strain-high-pass-freq $(HIGH_PASS_FREQ) \
                --frame-cache $^ \
                --max-high-freq-cutoff LRD \
                --approximant EOBNR \
                --order twoPN \
                --verbose \
                --spectrum-type LIGO \
                && ./prepareBank4IMRSA -N 15 V1-TMPLTBANK-871154847-2048.xml\
                && ./bank_splitter -i H1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) IMRSA-TMPLATEBANK-V1-TMPLTBANK-871154847-2048.xml \
                && ./bank_splitter -i V1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) IMRSA-TMPLATEBANK-V1-TMPLTBANK-871154847-2048.xml \
                && ./bank_splitter -i L1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -F -M $(HIGH_FREQUENCY_CUTOFF) IMRSA-TMPLATEBANK-V1-TMPLTBANK-871154847-2048.xml

mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz :
	./psd_xml_from_asd_txt H1 mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.txt

mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz : 
	./psd_xml_from_asd_txt H2 mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.txt

mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz : 
	./psd_xml_from_asd_txt L1 mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.txt

dag : H1-TMPLTBANK-871154847-2048.xml mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz
	ls *H1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *H2_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H2_split_bank.cache
	ls *L1_split_bank-H1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > L1_split_bank.cache
	./svd_bank_pipe H1 mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz H1_split_bank.cache
	./svd_bank_pipe H2 mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz H2_split_bank.cache
	./svd_bank_pipe L1 mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz L1_split_bank.cache

ninja : V1-TMPLTBANK-871154847-2048.xml ../psd/H1-871147452-871753660-reference_psd.xml.gz ../psd/L1-871147452-871753660-reference_psd.xml.gz ../psd/V1-871147452-871753660-reference_psd.xml.gz
	ls *H1_split_bank-*V1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *L1_split_bank-*V1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > L1_split_bank.cache
	ls *V1_split_bank-*V1-TMPLTBANK-871154847-2048.xml | lalapps_path2cache > V1_split_bank.cache
	./svd_bank_pipe H1 ../psd/H1-871147452-871753660-reference_psd.xml.gz H1_split_bank.cache
	./svd_bank_pipe L1 ../psd/L1-871147452-871753660-reference_psd.xml.gz L1_split_bank.cache
	./svd_bank_pipe V1 ../psd/V1-871147452-871753660-reference_psd.xml.gz V1_split_bank.cache


realclean :
	rm *.sub *.dag* *.cache *.sh H1-TMPLTBANK-871154847-2048.xml *split_bank-H1-TMPLTBANK-871154847-2048.xml mode_*.xml.gz
	rm -r logs
