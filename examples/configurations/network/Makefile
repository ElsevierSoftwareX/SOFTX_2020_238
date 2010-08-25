FILES = \
	H1-TMPLTBANK_LLOID_S6_ONLINE-958740096-2048.xml.gz \
	tmpltbank.xml.gz \
	reference_psd.H1.xml.gz \
	reference_psd.L1.xml.gz \
	reference_psd.V1.xml.gz \
	bank.H1.pickle \
	bank.L1.pickle \
	bank.V1.pickle \
	ligo_data_find.out \
	swinj.xml

all: $(FILES)

.PHONY: clean
clean:
	rm -f $(FILES)

# Injections go from July through October 1 00:00 UTC
swinj.xml:
	lalapps_inspinj \
		--output $@ \
		--user-tag LLOID_S6_ONLINE \
		--gps-start-time 961977615 \
		--gps-end-time 969926415 \
		--t-distr fixed \
		--time-step 900 \
		--d-distr volume \
		--min-distance 50e3 \
		--max-distance 75e3 \
		--l-distr random \
		--m-distr componentMass \
		--min-mass1 1.2 \
		--max-mass1 1.6 \
		--min-mass2 1.2 \
		--max-mass2 1.6 \
		--min-mtotal 2.4 \
		--max-mtotal 3.2 \
		--i-distr uniform \
		--waveform GeneratePPNtwoPN \
		--f-lower 10.0 \
		--disable-spin \
		--verbose

ligo_data_find.out:
	ligo_data_find -o H -t H1_DMT_C00_L2 -u file -l -s 958739936 -e 958743552 > $@

H1-TMPLTBANK_LLOID_S6_ONLINE-958740096-2048.xml.gz: ligo_data_find.out
	lalapps_tmpltbank \
		--verbose \
		--user-tag LLOID_S6_ONLINE \
		--gps-start-time 958740096 \
		--gps-end-time 958742144 \
		--grid-spacing Hexagonal \
		--dynamic-range-exponent 69.0 \
		--enable-high-pass 30.0 --high-pass-order 8 \
		--strain-high-pass-order 8 \
		--minimum-mass 1.2 --maximum-mass 1.6 \
		--approximant TaylorF2 --order twoPN \
		--standard-candle \
		--calibrated-data real_8 \
		--candle-mass1 1.4 --candle-mass2 1.4 \
		--channel-name H1:DMT-STRAIN --frame-cache ligo_data_find.out \
		--space Tau0Tau3 --number-of-segments 15 \
		--minimal-match 0.98 --candle-snr 8 --debug-level 33 --high-pass-attenuation 0.1 \
		--min-high-freq-cutoff SchwarzISCO --max-high-freq-cutoff SchwarzISCO \
		--segment-length 524288 \
		--low-frequency-cutoff 40.0 --pad-data 8 --num-freq-cutoffs 1 \
		--sample-rate 2048 --high-frequency-cutoff 921.6 --resample-filter ldas \
		--strain-high-pass-atten 0.1 --strain-high-pass-freq 30 \
		--min-total-mass 2.4 --max-total-mass 3.2 \
		--write-compress --spectrum-type median

tmpltbank.xml.gz: H1-TMPLTBANK_LLOID_S6_ONLINE-958740096-2048.xml.gz
	gstlal_prune_duplicate_mass_pairs $< $@

reference_psd.H1.xml.gz:
	gstlal_reference_psd \
		--verbose \
		--online-data \
		--instrument H1 \
		--write-psd $@ \
		--gps-start-time 958739939 \
		--gps-end-time 958743539

reference_psd.L1.xml.gz:
	gstlal_reference_psd \
		--verbose \
		--online-data \
		--instrument L1 \
		--write-psd $@ \
		--gps-start-time 958744974 \
		--gps-end-time 958747374

reference_psd.V1.xml.gz:
	gstlal_reference_psd \
		--verbose \
		--online-data \
		--instrument V1 \
		--write-psd $@ \
		--gps-start-time 966704415 \
		--gps-end-time 966711615

bank.%.pickle: reference_psd.%.xml.gz tmpltbank.xml.gz
	gstlal_svd_bank \
		--verbose \
		--template-bank tmpltbank.xml.gz \
		--ortho-gate-fap 0.01 \
		--reference-psd $< \
		--write-bank $@
