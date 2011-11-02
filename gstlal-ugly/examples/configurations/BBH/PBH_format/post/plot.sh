export GPSSTART=871147452
export GPSEND=871753660

for ifo in H1 V1 L1
do
    lalapps_sire \
        --injection-file HL-TEST2_INJECTIONS_2-871147452-604800.xml \
        --ifo-cut ${ifo}       \
        --cluster-time 4000    \
        --injection-window 100 \
        --output ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147452-604800.xml \
        --missed-injections ${ifo}-SIRE_MISSED_SUMMARY_FIRST-871147452-604800.xml \
        --summary-file ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147452-604800.txt \
        --data-type all_data \
        --cluster-algorithm snr	\
	--verbose \
	svd_${ifo}-INSPIRAL_LLOID.xml.gz

    ls -1 ${ifo}*FOUND*xml ${ifo}*MISSED*xml | lalapps_path2cache > foundmissed.cache

    plotinspmissed -a -b -c -d -e -j     \
        --ifo-times ${ifo} --ifo ${ifo}  \
        --user-tag FULL_DATA             \
        --output-path inspmissed_results \
        --found-pattern FOUND --missed-pattern MISSED \
        --cache-file foundmissed.cache \
        --sire \
        -O     \
        --gps-start-time ${GPSSTART} --gps-end-time ${GPSEND}

	./plot_recovered_params ${ifo} HL-TEST2_INJECTIONS_2-871147452-604800.xml highmass_test2_snr.xml ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147452-604800.xml > ${ifo}_plot_recovered_params_wiki_tables

done
