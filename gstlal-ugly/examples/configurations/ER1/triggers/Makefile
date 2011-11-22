H1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/H1_reference_psd_svd_bank.cache
H2_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/H2_reference_psd_svd_bank.cache
L1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/V1_reference_psd_svd_bank.cache
NUMBANKS = 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
#NUMBANKS = 1
MAXJOBS = 1400

all : dag

dag :
	gstlal_ll_trigger_pipe --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --num-banks $(NUMBANKS) --max-jobs $(MAXJOBS)

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite
