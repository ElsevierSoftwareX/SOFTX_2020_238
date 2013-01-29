trigtime=956498644
start=$(shell python -c "print ${trigtime} - 2000")
end=$(shell python -c "print ${trigtime} + 2000")
ep_start=$(shell python -c "print ${trigtime} - 200")
ep_end=$(shell python -c "print ${trigtime} + 200")

triggers: h1_darm_ep h1_accel_ep h1_seis_lvea_ep h1_seis_ey_ep
	ligolw_add *triggers* -o H_triggers_all.xml.gz
	rm *.xml

H1_seismic.cache:
	ligo_data_find -o H -t R -s ${start} -e ${end} -u file -l > H1_seismic.cache

h1_darm_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames \
		--initialization-file gstlal_excesspower_darm.ini  \
		--frame-cache H1_seismic.cache \
		--gps-start-time 956498544 \
		--gps-end-time 956498744 \
		--channel-name H1=LSC-DARM_ERR \
		--stream-tfmap "darm" \
		--sample-rate 2048

h1_accel_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames \
		--initialization-file gstlal_excesspower_accel.ini  \
		--frame-cache H1_seismic.cache \
		--gps-start-time 956498544 \
		--gps-end-time 956498744 \
		--channel-name H1=PEM-ISCT10_ACCY \
		--stream-tfmap "accel" \
		--sample-rate 1024

h1_seis_lvea_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_seismic.ini  \
		--frame-cache H1_seismic.cache \
		--gps-start-time 956497644 \
		--gps-end-time 956499644 \
		--channel-name H1=PEM-LVEA_SEISZ \
		--stream-tfmap "lvea_seisz" \
		--sample-rate 256

h1_seis_ey_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_seismic.ini  \
		--frame-cache H1_seismic.cache \
		--gps-start-time 956497644 \
		--gps-end-time 956499644 \
		--channel-name H1=PEM-EY_SEISZ \
		--stream-tfmap "lvea_ey" \
		--sample-rate 256

clean:
	rm -rf *.xml *.xml.gz *.cache *.ogg
