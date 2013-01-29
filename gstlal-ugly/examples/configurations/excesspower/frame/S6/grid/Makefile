trigtime=`lalapps_tconvert Aug 4 2010 6:35 UTC`
start=$(shell python -c "print ${trigtime} - 2000")
end=$(shell python -c "print ${trigtime} + 2000")
ep_start=$(shell python -c "print ${trigtime} - 200")
ep_end=$(shell python -c "print ${trigtime} + 200")

triggers: h1_darm_ep h1_omc_ep
	ligolw_add *triggers* -o H_triggers_all.xml.gz
	rm *.xml

h1.cache:
	ligo_data_find -o H -t R -s ${start} -e ${end} -u file -l > H1_grid.cache

h1_darm_ep: h1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames \
		--initialization-file gstlal_excesspower_darm.ini  \
		--frame-cache h1.cache \
		--gps-start-time ${ep_start} \
		--gps-end-time ${ep_end} \
		--stream-tfmap "darm" \
		--sample-rate 2048

h1_omc_ep: h1.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_darm.ini  \
		--frame-cache h1.cache \
		--gps-start-time ${ep_start} \
		--gps-end-time ${ep_end} \
		--channel-name H1=OMC-QPD3_SUM_IN1_DAQ \
		--stream-tfmap "omc" \
		--sample-rate 2048

clean:
	rm -rf *.xml *.xml.gz *.cache *.ogg
