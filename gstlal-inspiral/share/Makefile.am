dist_pkgdata_DATA = \
	inspiral_snr_pdf.xml.gz \
	ll_simplify_and_cluster.sql \
	ll_simplify.sql \
	Makefile.triggers_example \
	Makefile.non_spinning_NSBH \
	Makefile.non_spinning_BNS \
	Makefile.spinning_NSBH \
	Makefile.spinning_BNS \
	Makefile.offline_analysis_rules \
	Makefile.online_analysis \
	Makefile.online_bank \
	Makefile.IMBH_example \
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
