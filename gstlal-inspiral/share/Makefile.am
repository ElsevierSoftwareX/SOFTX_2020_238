if COND_MASSMODEL
MASSMODELS = population_models
else
MASSMODELS =
endif

dist_pkgdata_DATA = \
	de_calc_likelihood.sql \
	G1K1_p_of_instruments_given_H_d.h5 \
	H1K1_p_of_instruments_given_H_d.h5 \
	H1K1L1_p_of_instruments_given_H_d.h5 \
	H1K1V1_p_of_instruments_given_H_d.h5 \
	H1K1L1V1_p_of_instruments_given_H_d.h5 \
	H1L1_p_of_instruments_given_H_d.h5 \
	H1L1V1_p_of_instruments_given_H_d.h5 \
	H1V1_p_of_instruments_given_H_d.h5 \
	inj_simplify_and_cluster.sql \
	inj_snr_simplify_and_cluster.sql \
	inspiral_datalesslndensity.xml.gz \
	inspiral_dtdphi_pdf.h5 \
	inspiral_snr_pdf.xml.gz \
	K1L1_p_of_instruments_given_H_d.h5 \
	K1L1V1_p_of_instruments_given_H_d.h5 \
	K1V1_p_of_instruments_given_H_d.h5 \
	L1V1_p_of_instruments_given_H_d.h5 \
	ll_simplify.sql \
	ll_simplify_and_cluster.sql \
	Makefile.offline_analysis_rules \
	O1_snr_chi_pdf_reference.xml.gz \
	simplify_and_cluster.sql \
	simplify_proc_table_in_inj_file.sql \
	snr_simplify_and_cluster.sql

inspiral_snr_pdf.xml.gz :
	gstlal_inspiral_make_snr_pdf --verbose --output $@ \
		--horizon-distances H1=0,L1=0 \
		--horizon-distances H1=0,L1=100 \
		--horizon-distances H1=100,L1=0 \
		--horizon-distances H1=100,L1=100 \
		--min-instruments 1

SUBDIRS = $(MASSMODELS)
