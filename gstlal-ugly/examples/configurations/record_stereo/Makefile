# Run as:
# make -j CANDIDATE_TIME=<<gps time in seconds>>

REF_PSD_START_TIME = $(shell python -c "print $(CANDIDATE_TIME) - 500")
REF_PSD_END_TIME = $(shell python -c "print $(CANDIDATE_TIME) + 500")

FILES = \
	reference_psd.H1.xml.gz \
	reference_psd.L1.xml.gz \
	stereo_H1_L1.wav

all: $(FILES)

clean:
	rm -f $(FILES)

stereo_H1_L1.wav: reference_psd.H1.xml.gz reference_psd.L1.xml.gz encode.py
	./encode.py --gps-time $(CANDIDATE_TIME)

reference_psd.%.xml.gz:
	gstlal_reference_psd \
		--verbose \
		--online-data \
		--instrument $* \
		--write-psd $@ \
		--gps-start-time $(REF_PSD_START_TIME) \
		--gps-end-time $(REF_PSD_END_TIME)

