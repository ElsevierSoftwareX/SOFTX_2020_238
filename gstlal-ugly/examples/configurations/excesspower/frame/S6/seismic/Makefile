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
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_darm.ini  \
		--channel-width 32  \
		--gps-start 956498544 \
		--gps-end 956498744 \
		--stream-tfmap "darm" \
		--sample-rate 2048

h1_accel_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_accel.ini  \
		--channel-width 32  \
		--gps-start 956498544 \
		--gps-end 956498744 \
		--stream-tfmap "accel" \
		--sample-rate 1024

h1_seis_lvea_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_seismic.ini  \
		--channel-width 32  \
		--gps-start 956497644 \
		--gps-end 956499644 \
		--stream-tfmap "lvea_seisz" \
		--sample-rate 256

h1_seis_ey_ep: H1_seismic.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source gwffile  \
		--initialization-file gstlal_excesspower_seismic.ini  \
		--channel-width 32  \
		--gps-start 956497644 \
		--gps-end 956499644 \
		--channel-name "PEM-EY_SEISZ" \
		--stream-tfmap "lvea_ey" \
		--sample-rate 256

clean:
	rm -rf *.xml *.xml.gz *.cache *.ogg
