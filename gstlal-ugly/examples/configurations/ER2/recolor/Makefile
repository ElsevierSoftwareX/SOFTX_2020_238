GPS_START=1016532000
#GPS_END=1016567488
GPS_END=$(shell echo "${GPS_START} + 1000" | bc)
ONLINE_DET=H1
ONLINE_CHANNEL=FAKE-STRAIN
CHANNEL=PSL-ISS_PDB_OUT_DQ
NEW_CHANNEL=FAKE-STRAIN
IN_SPEC=ZERO_DET_high_P.xml
SAMPLE_RATE=16384

all: recolor analyze recolored_psd.xml ${CHANNEL}_psd.xml

H2_PSL_channels.cache:
	ligo_data_find -o H -t PSL_RDS_A6 -s ${GPS_START} -e ${GPS_END} -u file -l > H2_PSL_channels.cache
	# Necessary to get recoloring pipeline to discover the files
	sed -i "s|\ PSL_RDS_A6| H2_PSL_RDS_A6|g" H2_PSL_channels.cache

recolored.cache: 
	find H2-${CHANNEL} -name "*.gwf" | lalapps_path2cache > recolored.cache

${IN_SPEC}: ZERO_DET_high_P.txt 
	gstlal_psd_xml_from_asd_txt --invert --instrument H1 --output H1.xml ZERO_DET_high_P.txt
	gstlal_psd_xml_from_asd_txt --invert --instrument H2 --output H2.xml ZERO_DET_high_P.txt
	gstlal_psd_xml_from_asd_txt --invert --instrument L1 --output L1.xml ZERO_DET_high_P.txt
	ligolw_add --output ${IN_SPEC} {H1,H2,L1}.xml
	rm {H1,H2,L1}.xml

recolor: H2_PSL_channels.cache ${IN_SPEC}
	gstlal_recolor_frames \
		--gps-start-time ${GPS_START} \
		--gps-end-time 1016533000 \
		--instrument H2 \
		--channel-name "${CHANNEL}" \
		--output-channel-name "${NEW_CHANNEL}" \
		--frame-cache H2_PSL_channels.cache \
		--recolor-psd ${IN_SPEC} \
		--track-psd \
		--sample-rate ${SAMPLE_RATE} \
		--verbose

recolor_notrack: H2_PSL_channels.cache ${IN_SPEC} ${CHANNEL}_psd.xml
	gstlal_recolor_frames \
		--gps-start-time ${GPS_START} \
		--gps-end-time 1016533000 \
		--instrument H2 \
		--channel-name "${CHANNEL}" \
		--output-channel-name "${NEW_CHANNEL}" \
		--frame-cache H2_PSL_channels.cache \
		--recolor-psd ${IN_SPEC} \
		--reference-psd ${CHANNEL}_psd.xml \
		--sample-rate ${SAMPLE_RATE} \
		--verbose

recolor_online: ${IN_SPEC} 
	gstlal_recolor_frames \
		--instrument ${ONLINE_DET} \
		--online-data \
		--channel-name "${ONLINE_CHANNEL}" \
		--output-channel-name "${NEW_CHANNEL}" \
		--recolor-psd ${IN_SPEC} \
		--sample-rate ${SAMPLE_RATE} \
		--verbose

analyze: recolored.cache
	gstlal_excesspower  \
		--data-source gwffile \
		--initialization-file gstlal_excesspower_psl.ini  \
		--sample-rate ${SAMPLE_RATE}  \
		--gps-start $(( ${GPS_START} + 1 )) \
		--gps-end 1016532900 \
		--channel-name "${CHANNEL}" \
		--channel-width 32 \
		--verbose
	ligolw_add *triggers*.xml -o H2_${CHANNEL}_triggers.xml.gz
	rm *triggers*.xml

recolored_psd.xml: recolored.cache 
	gstlal_reference_psd  \
		--frame-cache recolored.cache  \
		--write-psd recolored_psd.xml \
		--gps-start $(( ${GPS_START} + 1 )) \
		--gps-end 1016532900  \
		--instrument H2 \
		--channel-name "${CHANNEL}" \
		--sample-rate ${SAMPLE_RATE}  \
		--verbose

${CHANNEL}_psd.xml: H2_PSL_channels.cache
	gstlal_reference_psd  \
		--frame-cache H2_PSL_channels.cache  \
		--write-psd ${CHANNEL}_psd.xml \
		--gps-start ${GPS_START} \
		--gps-end ${GPS_END} \
		--instrument H2 \
		--channel-name "${CHANNEL}" \
		--sample-rate ${SAMPLE_RATE}  \
		--verbose

clean:
	rm -rf H2-${CHANNEL}/ H2_PSL_channels.cache recolored.cache *.xml*
