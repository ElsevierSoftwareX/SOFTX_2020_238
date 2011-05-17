PSD_CACHE = ../psd/psd_pipe.cache
H1_FRAME_CACHE = ../segments/frame.H1.cache
H2_FRAME_CACHE = ../segments/frame.H2.cache
L1_FRAME_CACHE = ../segments/frame.L1.cache
H1_VETOES = ../segments/H1vetoes.xml
H2_VETOES = ../segments/H2vetoes.xml
L1_VETOES = ../segments/L1vetoes.xml
INJECTION_FILE = ../segments/injections.xml
H1_BANK_CACHE = ../bank/mode_H1-TMPLTBANK-869040003-2048_svd_bank.cache
H2_BANK_CACHE = ../bank/mode_H2-TMPLTBANK-875211248-2048_svd_bank.cache
L1_BANK_CACHE = ../bank/mode_L1-TMPLTBANK-858087370-2048_svd_bank.cache 

all : dag

dag :
	./trigger_pipe --psd-cache $(PSD_CACHE) --frame-cache H1=$(H1_FRAME_CACHE),H2=$(H2_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes H1=$(H1_VETOES),H2=$(H2_VETOES),L1=$(L1_VETOES)

injections :
	./trigger_pipe --injections $(INJECTION_FILE) --psd-cache $(PSD_CACHE) --frame-cache H1=$(H1_FRAME_CACHE),H2=$(H2_FRAME_CACHE),L1=$(L1_FRAME_CACHE) --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes H1=$(H1_VETOES),H2=$(H2_VETOES),L1=$(L1_VETOES)
	
realclean :
	rm -r *.sub *.dag* *.cache *.sh logs
