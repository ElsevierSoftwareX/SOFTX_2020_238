START = 871147452
STOP =  871247452
IFOS = H

# Template bank parameters
MIN_MASS = 0.2
MAX_MASS = 1.0
MIN_TOTAL_MASS = $(shell echo 2 \* $(MIN_MASS) | bc)
MAX_TOTAL_MASS = $(shell echo 2 \* $(MAX_MASS) | bc)
LOW_FREQUENCY_CUTOFF = 40.0
HIGH_PASS_FREQ = 30.0
HIGH_FREQUENCY_CUTOFF = 2047.0
SAMPLE_RATE = 4096
NUM_SPLIT_TEMPLATES = 250
OVERLAP = 50
MM = 0.97

all : dag plots

lalapps_path2cache :
	wget 'http://www.lsc-group.phys.uwm.edu/cgit/lalsuite/plain/pylal/bin/lalapps_path2cache'
	chmod +x lalapps_path2cache

bank : 
	gstlal_bank_splitter --overlap $(OVERLAP) --instrument H1 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) V1-SBANK-871154847-2048.xml.gz \
	&& gstlal_bank_splitter --overlap $(OVERLAP) --instrument H2 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) V1-SBANK-871154847-2048.xml.gz \
	&& gstlal_bank_splitter --overlap $(OVERLAP) --instrument L1 --n $(NUM_SPLIT_TEMPLATES) --sort-by mchirp --add-f-final --max-f-final $(HIGH_FREQUENCY_CUTOFF) V1-SBANK-871154847-2048.xml.gz

mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H1 --output mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.txt

mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument H2 --output mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.txt

mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz : 
	gstlal_psd_xml_from_asd_txt --instrument L1 --output mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.txt

reference_psd.xml.gz: mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz
	ligolw_add mode_H1-TMPLTBANK-869040003-2048.strainspec_asd_v4.xml.gz mode_H2-TMPLTBANK-875211248-2048.strainspec_asd_v4.xml.gz mode_L1-TMPLTBANK-858087370-2048.strainspec_asd_v4.xml.gz | gzip > reference_psd.xml.gz

sql :
	cat uniqueify_coinc_definer.sql uniqueify_time_slide.sql > simplify.sql
	cat simplify.sql cluster.sql > simplify_and_cluster.sql
	cat drop_sim.sql simplify_and_cluster.sql > inj_simplify_and_cluster.sql

dag : bank reference_psd.xml.gz sql lalapps_path2cache
	ls *H1_split_bank-V1-SBANK-871154847-2048.xml.gz | ./lalapps_path2cache > H1_split_bank.cache
	ls *H2_split_bank-V1-SBANK-871154847-2048.xml.gz | ./lalapps_path2cache > H2_split_bank.cache
	ls *L1_split_bank-V1-SBANK-871154847-2048.xml.gz | ./lalapps_path2cache > L1_split_bank.cache
	gstlal_inspiral_svd_bank_pipe --instrument H1 --reference-psd reference_psd.xml.gz --bank-cache H1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H1_bank
	gstlal_inspiral_svd_bank_pipe --instrument H2 --reference-psd reference_psd.xml.gz --bank-cache H2_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name H2_bank
	gstlal_inspiral_svd_bank_pipe --instrument L1 --reference-psd reference_psd.xml.gz --bank-cache L1_split_bank.cache --overlap $(OVERLAP) --flow $(LOW_FREQUENCY_CUTOFF) --output-name L1_bank
	cat H2_bank.dag H1_bank.dag L1_bank.dag > bank.dag

plots : reference_psd.xml.gz
	gstlal_plot_psd reference_psd.xml.gz psd_plot

realclean :
	rm -r *.sub *.dag* *.cache *.sh *split_bank-V1-SBANK-871154847-2048.xml.gz mode_*.xml.gz logs simplify.sql simplify_and_cluster.sql inj_simplify_and_cluster.sql
