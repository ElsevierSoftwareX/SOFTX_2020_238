export GPSSTART=871147552
export GPSEND=876357464

for ifo in H1 V1 L1
do
    lalapps_sire \
        --injection-file hlv-injections_all_1-871149786-4838400.xml \
        --ifo-cut ${ifo}       \
        --injection-window 1000 \
        --output ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147552-4838400.xml \
        --missed-injections ${ifo}-SIRE_MISSED_SUMMARY_FIRST-871147552-4838400.xml \
        --summary-file ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147552-4838400.txt \
        --data-type all_data \
    	--verbose \
    	H1L1V1-ALL-COMBINED*-LLOID.xml.gz

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

	./plot_recovered_params ${ifo} hlv-injections_all_1-871149786-4838400.xml hlv-snrs_all_1_fixed_v1-871149786-4838400.xml ${ifo}-SIRE_FOUND_SUMMARY_FIRST-871147552-4838400.xml > ${ifo}_plot_recovered_params_wiki_tables

done
