H1_BANK_CACHE = /home/channa/ER1/bank3/H1_reference_psd_svd_bank.cache
H2_BANK_CACHE = /home/channa/ER1/bank3/H2_reference_psd_svd_bank.cache
L1_BANK_CACHE = /home/channa/ER1/bank3/L1_reference_psd_svd_bank.cache
V1_BANK_CACHE = /home/channa/ER1/bank3/V1_reference_psd_svd_bank.cache
NUMBANKS = 4

all : dag

dag :
	gstlal_ll_trigger_pipe --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --num-banks $(NUMBANKS)

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml
