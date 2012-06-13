H1_BANK_CACHE = /home/shooper/scratch/IIR/IIR_ER1/subbank/H1_bank.cache
L1_BANK_CACHE = /home/shooper/scratch/IIR/IIR_ER1/subbank/L1_bank.cache
V1_BANK_CACHE = /home/shooper/scratch/IIR/IIR_ER1/subbank/V1_bank.cache
BURST_INJ = /home/pankow/ER1/inj_only/burst_injections.all.xml

NUMBANKS = 5
MAXJOBS = 10

all : dag

seed : marginalized_likelihood.xml.gz likelihood.xml.gz
	gstlal_ll_trigger_pipe --do-iir-pipeline --bank-cache H1=$(H1_BANK_CACHE),L1=$(L1_BANK_CACHE),V1=$(V1_BANK_CACHE) --num-banks $(NUMBANKS) --max-jobs $(MAXJOBS) --likelihood-file likelihood.xml.gz --channel G1=FAKE-STRAIN --channel=H2=FAKE-STRAIN --channel=H1=FAKE-STRAIN --channel=T1=FAKE-STRAIN --channel=V1=FAKE_h_16384Hz_4R --channel=L1=FAKE-STRAIN --gracedb-far-threshold 1e-100 --control-peak-time 4 --fir-stride 4 --copy-likelihoods --marginalized-likelihood-file marginalized_likelihood.xml.gz

dag :
	gstlal_ll_trigger_pipe --do-iir-pipeline --bank-cache H1=$(H1_BANK_CACHE),L1=$(L1_BANK_CACHE),V1=$(V1_BANK_CACHE) --num-banks $(NUMBANKS) --max-jobs $(MAXJOBS) --likelihood-file likelihood.xml.gz --channel G1=FAKE-STRAIN --channel=H2=FAKE-STRAIN --channel=H1=FAKE-STRAIN --channel=T1=FAKE-STRAIN --channel=V1=FAKE_h_16384Hz_4R --channel=L1=FAKE-STRAIN --gracedb-far-threshold 1e-3 --control-peak-time 4 --fir-stride 4  --gracedb-group CBC --marginalized-likelihood-file marginalized_likelihood.xml.gz

#vetoes.xml.gz :
#	gstlal_vetoes_from_burst_inj --default-duration 4 --output vetoes.xml.gz --padding 16 $(BURST_INJ)

likelihood.xml.gz :
	gstlal_inspiral_create_prior_diststats --instrument H1 --instrument L1 --instrument V1 --write-likelihood $@ --verbose

marginalized_likelihood.xml.gz : likelihood.xml.gz
	gstlal_inspiral_marginalize_likelihood --output $@ --verbose likelihood.xml.gz

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs *.xml *.gz *.sqlite *.txt
