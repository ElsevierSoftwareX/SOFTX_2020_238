etime=`lalapps_tconvert sept 16 2010 06:42:23 UTC`
start=$(shell python -c "print ${etime} - 50")
end=$(shell python -c "print ${etime} + 50")
output=$(shell pwd)/triggers/

triggers: v1_ep h1_ep l1_ep
	ligolw_add *triggers* -o HLV_triggers_all.xml.gz
	rm *.xml

package: ${output}
	find ${output} -name "*.xml" | lalapps_path2cache -a > triggers.cache
	ligolw_add --input-cache triggers.cache --output HLV_triggers_${start}_$(( ${end} - ${start} )).xml.gz --verbose --remove-input
	rm -rf ${output}

caches: l1.cache h1.cache v1.cache

cluster: 
	find ${output} -name "*.xml" | lalapps_path2cache -a > triggers.cache
	ligolw_bucluster --verbose --cluster-algorithm excesspower --program gstlal_excesspower -i triggers.cache
	rm triggers.cache

l1.cache:
	ligo_data_find -o L -t L1_LDAS_C02_L2 -s ${start} -e ${end}  -u file -l > l1.cache
h1.cache:
	ligo_data_find -o H -t H1_LDAS_C02_L2 -s ${start} -e ${end} -u file -l > h1.cache
v1.cache:
	ligo_data_find -o V -t HrecV2 -s ${start} -e ${end} -u file -l > v1.cache

v1_ep: v1.cache
	# Note the *required* "site" entry in the ini file for Virgo
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_v1_hoft.ini  \
		--gps-start ${start}  \
		--gps-end ${end}  \
		--sample-rate 2048  

h1_ep: h1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_h1_hoft.ini  \
		--gps-start ${start}  \
		--gps-end ${end}  \
		--sample-rate 2048  

l1_ep: l1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_l1_hoft.ini  \
		--gps-start ${start}  \
		--gps-end ${end}  \
		--sample-rate 2048  

clean:
	rm -rf *.xml *.xml.gz *.cache ${output}

