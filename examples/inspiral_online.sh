rm -f output_H1_bank*.xml

gstlal_inspiral \
	--online-data \
	--instrument "H1" \
	--channel-name "DMT-STRAIN" \
	--output "output_H1.xml" \
	--template-bank=banks/1-split_bank-H1-TMPLTBANK_DATAFIND-871157768-2048.xml.gz \
	--reference-psd "reference_psd.xml.gz" \
	--comment "lloid rrocks" \
	--verbose

