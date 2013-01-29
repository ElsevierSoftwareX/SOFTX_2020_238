trigtime=967875896
start=$(shell python -c "print ${trigtime} - 200")
end=$(shell python -c "print ${trigtime} + 200")
ep_start=$(shell python -c "print ${trigtime} - 100")
ep_end=$(shell python -c "print ${trigtime} + 10")

triggers: l1_darm_ep l1_omc_ep l1_omc2_ep
	ligolw_add *triggers* -o L_triggers_all.xml.gz
	rm *.xml

L1_flip.cache:
	ligo_data_find -o L -t R -s ${start} -e ${end} -u file -l > L1_flip.cache

l1_darm_ep: L1_flip.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--frame-cache L1_flip.cache  \
		--initialization-file gstlal_excesspower_darm.ini  \
		--gps-start-time ${ep_start} \
		--gps-end-time ${ep_end} \
		--channel-name L1=LSC-DARM_ERR \
		--stream-tfmap "darm" \
		--sample-rate 1024

l1_omc_ep: L1_flip.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--frame-cache L1_flip.cache  \
		--initialization-file gstlal_excesspower_darm.ini  \
		--gps-start-time ${ep_start} \
		--gps-end-time ${ep_end} \
		--channel-name L1=OMC-QPD3_SUM_IN1_DAQ \
		--stream-tfmap "omc" \
		--sample-rate 1024

l1_omc2_ep: L1_flip.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--frame-cache L1_flip.cache  \
		--initialization-file gstlal_excesspower_darm.ini  \
		--gps-start-time ${ep_start} \
		--gps-end-time ${ep_end} \
		--channel-name L1=OMC-QPD3_Y_OUT_DAQ \
		--stream-tfmap "omc2" \
		--sample-rate 1024

clean:
	rm -rf *.xml *.xml.gz *.cache *.ogg
