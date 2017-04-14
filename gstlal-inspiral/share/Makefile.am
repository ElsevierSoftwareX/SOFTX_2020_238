dist_pkgdata_DATA = \
	inspiral_snr_pdf.xml.gz \
	ll_simplify_and_cluster.sql \
	ll_simplify.sql \
	Makefile.offline_analysis_rules \
	de_calc_likelihood.sql \
	inj_simplify_and_cluster.sql \
	simplify_and_cluster.sql \
	O1_snr_chi_pdf_reference.xml.gz

inspiral_snr_pdf.xml.gz :
	gstlal_inspiral_make_snr_pdf --verbose --output $@ \
		--horizon-distances H1=0,L1=0 \
		--horizon-distances H1=0,L1=100 \
		--horizon-distances H1=100,L1=0 \
		--horizon-distances H1=100,L1=100 \
		--min-instruments 1
