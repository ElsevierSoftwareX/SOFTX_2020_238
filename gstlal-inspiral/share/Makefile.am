dist_pkgdata_DATA = \
	de_calc_likelihood.sql \
	inj_simplify_and_cluster.sql \
	inspiral_snr_pdf.xml.gz \
	ll_simplify.sql \
	ll_simplify_and_cluster.sql \
	Makefile.offline_analysis_rules \
	O1_snr_chi_pdf_reference.xml.gz \
	simplify_and_cluster.sql

inspiral_snr_pdf.xml.gz :
	gstlal_inspiral_make_snr_pdf --verbose --output $@ \
		--horizon-distances H1=0,L1=0 \
		--horizon-distances H1=0,L1=100 \
		--horizon-distances H1=100,L1=0 \
		--horizon-distances H1=100,L1=100 \
		--min-instruments 1
