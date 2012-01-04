H1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/H1_reference_psd_svd_bank.cache
H2_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/H2_reference_psd_svd_bank.cache
L1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = /home/channa/ER1/bank_10Hz_1_25M/V1_reference_psd_svd_bank.cache
# for tests you can use this instead
# NUMBANKS = 1
NUMBANKS = 3,4,5,6,7,8,9,10,11,12,13
# FOR PRODUCTION SET THIS TO NUMBER OF AVAILABLE NODES
# MAXJOBS = 150
MAXJOBS = 10

all : seed

seed : likelihood.xml.gz
	gstlal_ll_trigger_pipe --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --num-banks $(NUMBANKS) --max-jobs $(MAXJOBS) --likelihood-file likelihood.xml.gz --channel G1=FAKE-STRAIN --channel=H2=FAKE-STRAIN --channel=H1=FAKE-STRAIN --channel=T1=FAKE-STRAIN --channel=V1=FAKE-STRAIN --channel=L1=FAKE-STRAIN --gracedb-far-threshold 1e-100 --control-peak-time 4 --fir-stride 4 --copy-likelihoods

dag :
	gstlal_ll_trigger_pipe --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --num-banks $(NUMBANKS) --max-jobs $(MAXJOBS) --likelihood-file likelihood.xml.gz --channel G1=FAKE-STRAIN --channel=H2=FAKE-STRAIN --channel=H1=FAKE-STRAIN --channel=T1=FAKE-STRAIN --channel=V1=FAKE-STRAIN --channel=L1=FAKE-STRAIN --gracedb-far-threshold 1e-5 --control-peak-time 4 --fir-stride 4


likelihood.xml.gz :
	gstlal_inspiral_create_prior_diststats --write-likelihood likelihood.xml.gz --verbose

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite
