#!/bin/bash

if [ -z "$1" ]
then
	echo argument 1 must be ifo
	exit 1
fi

IFO=$1

rm -f output_${IFO}_online.sqlite
GST_DEBUG=3 gstlal_inspiral \
	--online-data \
	--instrument ${IFO} \
	--output "output_${IFO}_online.sqlite" \
	--gps-start-time `lalapps_tconvert now` \
	--template-bank=banks/1-split_bank-H1-TMPLTBANK_DATAFIND-871157768-2048.xml.gz \
	--reference-psd "reference_psd.xml.gz" \
	--write-pipeline "inspiral_${IFO}_online" \
	--comment "lloid rrocks" \
	--verbose 2>&1 | tee inspiral_${IFO}_online.log
