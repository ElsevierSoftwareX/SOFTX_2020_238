etime=1012071168
blocksize=$(shell echo "16384 * 8" | bc)
start=$(shell python -c "print ${etime} - 50")
end=$(shell python -c "print ${etime} + 50")
output=$(shell pwd)/triggers/
image=unicorn.png
ts_table=time_slide_table.xml

H_BTLWNB_unicorn_100.xml:
	# NOTE: You'll need to generate a zero lag time slide table for this
	lalapps_binj_pic -v -n 1e-29 --sample-rate 4096 -s 103 -l 16 -d 16 -H 100 -o H_BTLWNB_unicorn_100.xml ${image} -T ${ts_table}

no_inj:
	gstlal_excesspower  \
		--verbose  \
		--channel-name=H1=FAKE-STRAIN \
		--block-size ${blocksize} \
		--gps-start-time 0 \
		--gps-end-time 10000 \
		--file-cache \
		--file-cache-name H1_no_inj.cache \
		--data-source AdvLIGO  \
		--initialization-file gstlal_excesspower_h1_generic_noinj.ini  \
		--sample-rate 4096  

no_inj_movie:
	gstlal_excesspower  \
		--verbose  \
		--channel-name=H1=FAKE-STRAIN \
		--block-size ${blocksize} \
		--gps-start-time 0 \
		--data-source AdvLIGO  \
		--disable-triggers \
		--stream-tfmap video \
		--initialization-file gstlal_excesspower_h1_generic_noinj.ini  \
		--sample-rate 4096  

unicorns: H_BTLWNB_unicorn_100.xml 
	gstlal_excesspower  \
		--verbose  \
		--channel-name=H1=FAKE-STRAIN \
		--block-size ${blocksize} \
		--gps-start-time 0 \
		--data-source AdvLIGO  \
		--injections H_BTLWNB_unicorn_100.xml \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

unicorn_movie: H_BTLWNB_unicorn_100.xml
	gstlal_excesspower  \
		--verbose  \
		--channel-name=H1=FAKE-STRAIN \
		--block-size ${blocksize} \
		--gps-start-time 0 \
		--data-source AdvLIGO  \
		--stream-tfmap unicorn  \
		--disable-triggers  \
		--injections H_BTLWNB_unicorn_100.xml \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

clean:
	rm -rf *triggers*.xml *level*.xml *.xml.gz *.cache ${output}

