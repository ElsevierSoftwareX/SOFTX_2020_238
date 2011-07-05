START = 871147452
STOP =  871247452
PSD_FILE = ../bank/reference_psd.xml.gz
FRAME_CACHE = ../segments/frame.cache
FRAME_SEGMENTS_FILE = ../segments/segments.xml
FRAME_SEGMENTS_NAME = RESULT
VETOES = ../segments/vetoes.xml
INJECTION_FILE = ../segments/injections.xml
H1_BANK_CACHE = ../bank/mode_H1-TMPLTBANK-869040003-2048_svd_bank.cache
H2_BANK_CACHE = ../bank/mode_H2-TMPLTBANK-875211248-2048_svd_bank.cache
L1_BANK_CACHE = ../bank/mode_L1-TMPLTBANK-858087370-2048_svd_bank.cache 

all : dag

tisi.xml :
	ligolw_tisi --instrument=H1=0:0:0 --instrument=H2=0:0:0 --instrument=L1=0:3.14159:3.14159 tisi.xml

dag : tisi.xml
	./trigger_pipe --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAME_CACHE)  --frame-segments-file $(FRAME_SEGMENTS_FILE)  --frame-segments-name $(FRAME_SEGMENTS_NAME) --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes $(VETOES) --time-slide-file tisi.xml

injections :
	./trigger_pipe --injections $(INJECTION_FILE) --gps-start-time $(START) --gps-stop-time $(STOP) --reference-psd $(PSD_FILE) --frame-cache $(FRAME_CACHE)  --frame-segments-file $(FRAME_SEGMENTS_FILE)  --frame-segments-name $(FRAME_SEGMENTS_NAME) --bank-cache H1=$(H1_BANK_CACHE),H2=$(H2_BANK_CACHE),L1=$(L1_BANK_CACHE) --vetoes $(VETOES)

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs tisi.xml
