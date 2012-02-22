START = 871147552
STOP = 871752352 
#STOP = 876357464 

# File locations
PSD_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//psd/reference_psd.xml.gz
PSD_FILE_H1 = ../psd/H1_reference_psd.xml.gz
PSD_FILE_L1 = ../psd/L1_reference_psd.xml.gz
PSD_FILE_V1 = ../psd/V1_reference_psd.xml.gz
DELETE_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/simplify_and_delete_by_far.sql
CLUSTER_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/simplify_and_cluster.sql
INJ_SQL_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//banks_smaller/inj_simplify_and_cluster.sql
H1_BANK_CACHE = ../H1_svd_bank.cache
L1_BANK_CACHE = ../L1_svd_bank.cache
V1_BANK_CACHE = ../V1_svd_bank.cache
FRAMES = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format//frames/frames.cache
WEBDIR = /home/mafrei/public_html/NINJA2/TwoMonthSet/sbank
SEGMENTS_FILE = /home/mafrei/runDirectory/IMRSA/NINJA2/TwoMonthCan/PBH_format/psd/segments.xml.gz

# Adjusting these will change the number of injections OR change the time-step value smaller makes more injections (currently overlap.py will assume these gps times)
# START = 871147452
# STOP = 875753660
DUR=$(shell echo $(STOP) - $(START) | bc -l)

# Template bank parameters
TMPLT_APPROX = IMRPhenomB
TMPLT_NOISE_MODEL = iLIGOSRD
MIN_MASS_TMP = 1.0
MAX_MASS_TMP = 349.0
MIN_TOTAL_MASS_TMP = 20.0
MAX_TOTAL_MASS_TMP = 350.0
LOW_FREQUENCY_CUTOFF_TMP = 40.0
HIGH_PASS_FREQ_TMP = 30.0
HIGH_FREQUENCY_CUTOFF_TMP = 1023.0
SAMPLE_RATE_TMP = 2048
MM_TMP = 0.97
MAX_MRATIO_TMP = 8.0

#Banksim Injection parameters
INJ_APPROX = IMRPhenomB
INJ_PSD = LIGOIPsd
MIN_MASS_1 = 1.0
MAX_MASS_1 = 349.0
MIN_MASS_2 = 1.0
MAX_MASS_2 = 349.0
MIN_TOTAL_MASS = 20.0
MAX_TOTAL_MASS = 350.0
LOW_FREQUENCY_CUTOFF = 40.0
MIN_SPIN = 0.0
MAX_SPIN = 0.5
MIN_MRATIO = 1.0
MAX_MRATIO = 10.0
TIME_STEP = 40

#Other Parameters
CLIPPING = 0
NUM_SPLIT_TEMPLATES = 250

sbank: ninja

bankSim : match_hist

#BankSim Injection file creation
HL-INJECTIONS_2345_BANKSIM-$(START)-$(DUR).xml:
	lalapps_inspinj --i-distr uniform --m-distr componentMass --enable-spin   --gps-start-time $(START) --user-tag BANKSIM --f-lower $(LOW_FREQUENCY_CUTOFF) --seed 2345 --l-distr random --d-distr log10 --min-distance 1000 --waveform IMRPhenomBpseudoFourPN --gps-end-time $(STOP) --max-distance 1000000 --time-step $(TIME_STEP) --max-mtotal $(MAX_TOTAL_MASS) --min-mtotal $(MIN_TOTAL_MASS) --min-mass1 $(MIN_MASS_1) --min-mass2 $(MIN_MASS_2) --max-mass1 $(MAX_MASS_1) --max-mass2 $(MAX_MASS_2) --aligned --min-spin1 $(MIN_SPIN) --min-spin2 $(MIN_SPIN) --max-spin1 $(MAX_SPIN) --max-spin2 $(MAX_SPIN) --min-mratio $(MIN_MRATIO) --max-mratio $(MAX_MRATIO) --verbose

Culled_2_HL-INJECTIONS_2345_BANKSIM-$(START)-$(DUR).xml: HL-INJECTIONS_2345_BANKSIM-$(START)-$(DUR).xml
	python cullInjections HL-INJECTIONS_2345_BANKSIM-$(START)-$(DUR).xml

# Do bankSim
IMRPBvIMRPB.npz: Culled_2_HL-INJECTIONS_2345_BANKSIM-$(START)-$(DUR).xml V1-SBANK-871154847-2048.xml
	bankSim $(basename $@) $^ $(INJ_APPROX) $(TMPLT_APPROX) $(INJ_PSD) $(LOW_FREQUENCY_CUTOFF) --verbose

match_hist: IMRPBvIMRPB.npz
	plot_bankSim $<

# Make template bank
V1-SBANK-871154847-2048.xml:
	makeBank --mass1-min $(MIN_MASS_TMP) --mass1-max $(MAX_MASS_TMP) --mtotal-min $(MIN_TOTAL_MASS_TMP) --mtotal-max $(MAX_TOTAL_MASS_TMP) --max-mass-ratio $(MAX_MRATIO_TMP) --match-min $(MM_TMP) --flow $(LOW_FREQUENCY_CUTOFF_TMP) --approximant $(TMPLT_APPROX) --noise-model $(TMPLT_NOISE_MODEL)  --user-tag $(basename $@) --aligned-spin --instrument V1 --verbose

split_banks: V1-SBANK-871154847-2048.xml
	./gstlal_bank_splitter -i H1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -M $(HIGH_FREQUENCY_CUTOFF_TMP) -O 0  $^ 
	./gstlal_bank_splitter -i V1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -M $(HIGH_FREQUENCY_CUTOFF_TMP) -O 0  $^
	./gstlal_bank_splitter -i L1 -n $(NUM_SPLIT_TEMPLATES) -s mchirp -M $(HIGH_FREQUENCY_CUTOFF_TMP) -O 0  $^

ninja : split_banks
	ls *H1_split_bank-*V1-SBANK-871154847-2048.xml | lalapps_path2cache > H1_split_bank.cache
	ls *L1_split_bank-*V1-SBANK-871154847-2048.xml | lalapps_path2cache > L1_split_bank.cache
	ls *V1_split_bank-*V1-SBANK-871154847-2048.xml | lalapps_path2cache > V1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --instrument H1 --reference-psd $(PSD_FILE_H1) --output-name H1_svd_bank --bank-cache H1_split_bank.cache --stagger
	gstlal_inspiral_svd_bank_pipe --instrument L1 --reference-psd $(PSD_FILE_L1) --output-name L1_svd_bank --bank-cache L1_split_bank.cache --stagger
	gstlal_inspiral_svd_bank_pipe --instrument V1 --reference-psd $(PSD_FILE_V1) --output-name V1_svd_bank --bank-cache V1_split_bank.cache --stagger
