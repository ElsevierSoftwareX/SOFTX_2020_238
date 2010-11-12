all: aligo.dag H1-TMPLTBANK-1024-2048.xml.gz psd.xml.gz

aligo.dag: H1-TMPLTBANK-1024-2048.xml.gz psd.xml.gz
	./build_job $<

psd.xml.gz: frame.cache
	gstlal_reference_psd \
		--verbose \
		--instrument H1 \
		--channel-name MOCK-STRAIN \
		--gps-start-time 1024 \
		--gps-end-time 3073 \
		--frame-cache $^ \
		--write-psd $@
	
H1-TMPLTBANK-1024-2048.xml.gz: frame.cache
	lalapps_tmpltbank \
		--disable-compute-moments \
		--grid-spacing Hexagonal \
		--dynamic-range-exponent 69.0 \
		--enable-high-pass 5.0 \
		--high-pass-order 8 \
		--strain-high-pass-order 8 \
		--minimum-mass 1.0 \
		--maximum-mass 3.0 \
		--min-total-mass 2.0 \
		--max-total-mass 6.0 \
		--gps-start-time 1024 \
		--gps-end-time 3072 \
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1 \
		--candle-mass2 1 \
		--channel-name H1:MOCK-STRAIN \
		--space Tau0Tau3 \
		--number-of-segments 127 \
		--minimal-match 0.97 \
		--candle-snr 8 \
		--debug-level 33 \
		--high-pass-attenuation 0.1 \
		--min-high-freq-cutoff SchwarzISCO \
		--segment-length 131072 \
		--low-frequency-cutoff 10.0 \
		--pad-data 8 \
		--num-freq-cutoffs 1 \
		--sample-rate 4096 \
		--high-frequency-cutoff 2048.0 \
		--resample-filter ldas \
		--strain-high-pass-atten 0.1 \
		--strain-high-pass-freq 5 \
		--frame-cache $^ \
		--max-high-freq-cutoff SchwarzISCO \
		--approximant TaylorF2 \
		--write-compress \
		--order twoPN \
		--spectrum-type median

frame.cache:
	./glob_data_find $(PWD)/hoft > $@

clean:
	rm -f frame.cache *.sub *.dag *.H1-TMPLTBANK-1024-2048.xml.gz *.H1-TMPLTBANK-1024-2048.xml.gz.* H1-TMPLTBANK-1024-2048.xml.gz psd.xml.gz
