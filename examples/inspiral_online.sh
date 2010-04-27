rm -f output_H1_bank*.xml

GST_DEBUG=3 gstlal_inspiral \
	--online-data \
	--instrument "H1" \
	--output "output_H1.xml" \
	--gps-start-time `lalapps_tconvert now` \
	--template-bank=banks/1-split_bank-H1-TMPLTBANK_DATAFIND-871157768-2048.xml.gz \
	--reference-psd "reference_psd.xml.gz" \
	--comment "lloid rrocks" \
	--verbose
