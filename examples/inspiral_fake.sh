rm -f output_H1_fake.sqlite

gstlal_inspiral \
	--fake-data \
	--instrument "H1" \
	--output "output_H1_fake.xml" \
	--gps-start-time 957115711 \
	--template-bank=banks/1-split_bank-H1-TMPLTBANK_DATAFIND-871157768-2048.xml.gz \
	--reference-psd "reference_psd.xml.gz" \
	--write-pipeline "inspiral_fake" \
	--comment "lloid rrocks" \
	--verbose 2>&1 | tee inspiral_fake.log
