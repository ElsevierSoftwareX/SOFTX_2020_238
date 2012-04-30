etime=1012071168
start=$(shell python -c "print ${etime} - 50")
end=$(shell python -c "print ${etime} + 50")
output=$(shell pwd)/triggers/

H_BTLWNB_unicorn_100.xml:
	lalsim-unicorn -d H -H 1e-21 -s 4096 -t 100 > H_BTLWNB_unicorn_100.xml

no_inj:
	gstlal_excesspower  \
		--verbose  \
		--data-source fakeadvLIGO  \
		--initialization-file gstlal_excesspower_h1_generic_noinj.ini  \
		--sample-rate 4096  

unicorns: H_BTLWNB_unicorn_100.xml 
	gstlal_excesspower  \
		--verbose  \
		--data-source fakeadvLIGO  \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

clean:
	rm -rf *triggers*.xml *level*.xml *.xml.gz *.cache ${output}

