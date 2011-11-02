PSD_CACHE = ../psd/psd_pipe.cache
H1_FRAME_CACHE = ../segments/frame.H1.cache
V1_FRAME_CACHE = ../segments/frame.V1.cache
L1_FRAME_CACHE = ../segments/frame.L1.cache
#H1_VETOES = /dev/null
#V1_VETOES = /dev/null
#L1_VETOES = /dev/null
INJECTION_FILE = /dev/null
H1_BANK_CACHE = ../bank/H1_svd_bank.cache
V1_BANK_CACHE = ../bank/V1_svd_bank.cache
L1_BANK_CACHE = ../bank/L1_svd_bank.cache

all : dag

dag :
	./trigger_pipe --psd-cache $(PSD_CACHE) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --channel LDAS-STRAIN
#	./trigger_pipe --psd-cache $(PSD_CACHE) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes H1=$(H1_VETOES),V1=$(V1_VETOES),L1=$(L1_VETOES)

injections :
	./trigger_pipe --injections $(INJECTION_FILE) --psd-cache $(PSD_CACHE) --frame-cache H1=$(H1_FRAME_CACHE),V1=$(V1_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --bank-cache H1=$(H1_BANK_CACHE),V1=$(V1_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes H1=$(H1_VETOES),V1=$(V1_VETOES),L1=$(L1_VETOES)
	
realclean :
	rm *.sub *.dag* *.cache *.sh
	rm -r logs
