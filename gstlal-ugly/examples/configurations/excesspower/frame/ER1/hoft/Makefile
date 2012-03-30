etime=1012071168
start=$(shell python -c "print ${etime} - 50")
end=$(shell python -c "print ${etime} + 50")

triggers: v1_ep h1_ep l1_ep
	ligolw_add *triggers* -o HLV_triggers_all.xml.gz
	rm *.xml

caches: l1.cache h1.cache h2.cache v1.cache

l1.cache:
	ligo_data_find -o L -t L1_DMT_ERHOFT -s ${start} -e ${end}  -u file -l > l1.cache
h1.cache:
	ligo_data_find -o H -t H1H2_DMT_ERHOFT -s ${start} -e ${end} -u file -l > h1.cache
h2.cache:
	ligo_data_find -o H -t H1H2_DMT_ERHOFT -s ${start} -e ${end} -u file -l > h2.cache
v1.cache:
	ligo_data_find -o V -t ER1_hoft -s ${start} -e ${end} -u file -l > v1.cache

v1_ep: caches
	# Note the *required* "site" entry in the ini file for Virgo
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_v1_hoft.ini  \
		--gps-start 1012071068 \
		--gps-end 1012071268 \
		--sample-rate 2048  

h1_ep: caches
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_h1_hoft.ini  \
		--sample-rate 2048  

l1_ep: caches
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_l1_hoft.ini  \
		--sample-rate 2048  

clean:
	rm -rf *.xml *.xml.gz *.cache

