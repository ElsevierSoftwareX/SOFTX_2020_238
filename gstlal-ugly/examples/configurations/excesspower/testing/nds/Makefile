
ndstest_L1:
	gstlal_excesspower  \
		--verbose  \
		--data-source nds  \
		--channel-name L1=PEM-EY_SEISX_OUT_DQ \
		--nds-host llocds.ligo-la.caltech.edu \
		--nds-port 31200 \
		--initialization-file gstlal_excesspower_l1_generic.ini  \
		--sample-rate 4096  

ndstest_L1_stream:
	gstlal_excesspower  \
		--verbose  \
		--data-source nds  \
		--stream-tfmap video \
		--disable-triggers \
		--channel-name L1=PEM-EY_SEISX_OUT_DQ \
		--initialization-file gstlal_excesspower_l1_generic.ini  \
		--sample-rate 4096  

ndstest_L1_PSL_stream:
	gstlal_excesspower  \
		--verbose  \
		--data-source nds  \
		--stream-tfmap video \
		--disable-triggers \
		--initialization-file gstlal_excesspower_l1_generic.ini  \
		--channel-name L1=PSL-ISS_PDA_OUT_DQ \
		--sample-rate 4096  

ndstest_H1:
	gstlal_excesspower  \
		--verbose  \
		--data-source nds  \
		--initialization-file gstlal_excesspower_h1_generic.ini  \
		--sample-rate 4096  

clean:
	rm -rf *triggers*.xml *level*.xml *.xml.gz *.cache ${output}

