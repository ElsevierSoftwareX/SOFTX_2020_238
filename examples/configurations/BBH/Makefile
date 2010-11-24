START = 871147452
STOP = 871753660
IFOS = H V L

all : dag

segments.%.xml:
	ligo_data_find -o $* -t $*1_NINJA_HIGHMASS_TEST2 -l  -s $(START) -e $(STOP) --url-type file > frame.$*.cache
	./cache_to_segments frame.$*.cache segments.$*.xml

V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz: frame.V.cache 
	lalapps_tmpltbank --disable-compute-moments --grid-spacing Hexagonal --dynamic-range-exponent 6.900000e+01 --enable-high-pass 3.000000e+01 --high-pass-order 8 --strain-high-pass-order 8 --maximum-mass 9.900000e+01 --approximant EOBNR --gps-end-time 871149508 --calibrated-data real_4 --candle-mass1 1.400000e+00 --candle-mass2 1.400000e+00 --channel-name V1:LDAS-STRAIN --space Tau0Tau3 --number-of-segments 15 --minimal-match 9.700000e-01 --candle-snr 8.000000e+00 --debug-level 33 --gps-start-time 871147460 --high-pass-attenuation 1.000000e-01 --min-high-freq-cutoff LRD --segment-length 524288 --low-frequency-cutoff 4.000000e+01 --pad-data 8 --num-freq-cutoffs 1 --sample-rate 2048 --high-frequency-cutoff 1.023000e+03 --resample-filter ldas --strain-high-pass-atten 1.000000e-01 --strain-high-pass-freq 3.000000e+01 --min-total-mass 2.500000e+01 --max-total-mass 1.000000e+02 --frame-cache frame.V.cache --max-high-freq-cutoff LRD --user-tag DATAFIND --minimum-mass 1.000000e+00 --order twoPN --spectrum-type median --write-compress  --standard-candle --verbose && ./prepareBank4IMRSA -N 15 V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz && ./bank_splitter -n 500 -s ffinal -F IMRSA-TMPLATEBANK-V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz \

segments.xml : $(IFOS:%=segments.%.xml)
	ligolw_add --output segments.xml segments.H.xml segments.L.xml segments.V.xml

dag : segments.xml V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz
	for f in `ls -I 'DAG*' -d *split_bank-IMRSA-TMPLATEBANK-V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz`; do \
		mkdir -p DAG_$${f}; \
		cp gstlal_inspiral.ini DAG_$${f} ;\
		sed --in-place s@bank_file@../$${f}@g DAG_$${f}/gstlal_inspiral.ini ;\
		pushd DAG_$${f} && rm -f vetoes.xml.gz ; ligolw_segments --verbose --output vetoes.xml.gz --name vetoes --insert-from-segwizard={H1,L1,V1}=/dev/null && ligolw_tisi --verbose --instrument L1=0:0:0 --instrument H1=0:0:0 --instrument V1=0:0:0 time_slides.xml.gz && gstlal_inspiral_pipe --verbose --config-file gstlal_inspiral.ini --log-path $(TMPDIR) --time-slides time_slides.xml.gz --vetoes vetoes.xml.gz ; popd; \
	done

realclean :
	for n in `ls -d DAG_*` ; do \
		rm -Rvf $${n} ; \
	done
	rm -r segments*xml frame*cache *split_bank-IMRSA-TMPLATEBANK-V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz IMRSA-TMPLATEBANK-V1-TMPLTBANK_DATAFIND-871147460-2048.xml.gz time_slides.xml.gz vetoes.xml.gz
