etime=`lalapps_tconvert sept 16 2010 06:42:23 UTC`
start=$(shell python -c "print ${etime} - 300")
end=$(shell python -c "print ${etime} + 50")
output=$(shell pwd)/triggers/

triggers: v1_ep h1_ep l1_ep
	ligolw_add H1-*.xml L1-*.xml V1-*.xml -o HLV_triggers_all.xml.gz
	rm *.xml

package: ${output}
	find ${output} -name "*.xml" | lalapps_path2cache -a > triggers.cache
	ligolw_add --input-cache triggers.cache --output HLV_triggers_${start}_$(( ${end} - ${start} )).xml.gz --verbose --remove-input
	rm -rf ${output}

caches: l1.cache h1.cache v1.cache

cluster: 
	ligolw_bucluster --verbose --cluster-algorithm excesspower --program gstlal_excesspower HLV_triggers_all.xml.gz

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
		--data-source frames  \
		--initialization-file gstlal_excesspower_v1_hoft.ini  \
		--frame-cache v1.cache \
		--gps-start-time ${start}  \
		--gps-end-time ${end}  \
		--channel-name V1=h_16384Hz \
		--sample-rate 2048  

h1_ep: h1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames \
		--initialization-file gstlal_excesspower_h1_hoft.ini  \
		--frame-cache h1.cache \
		--gps-start-time ${start}  \
		--gps-end-time ${end}  \
		--channel-name H1=LDAS-STRAIN \
		--sample-rate 2048  

l1_ep: l1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames \
		--initialization-file gstlal_excesspower_l1_hoft.ini  \
		--frame-cache l1.cache \
		--gps-start-time ${start}  \
		--gps-end-time ${end}  \
		--channel-name L1=LDAS-STRAIN \
		--sample-rate 2048  

clean:
	rm -rf *.xml *.xml.gz *.cache ${output}

