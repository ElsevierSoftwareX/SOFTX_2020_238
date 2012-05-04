etime=1012071168
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
		--data-source fakeadvLIGO  \
		--initialization-file gstlal_excesspower_h1_generic_noinj.ini  \
		--sample-rate 4096  

unicorns: H_BTLWNB_unicorn_100.xml 
	./gstlal_excesspower  \
		--verbose  \
		--data-source fakeadvLIGO  \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

unicorn_movie: H_BTLWNB_unicorn_100.xml
	./gstlal_excesspower  \
		--verbose  \
		--data-source fakeadvLIGO  \
		--stream-tfmap unicorn  \
		--disable-triggers  \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

clean:
	rm -rf *triggers*.xml *level*.xml *.xml.gz *.cache ${output}

