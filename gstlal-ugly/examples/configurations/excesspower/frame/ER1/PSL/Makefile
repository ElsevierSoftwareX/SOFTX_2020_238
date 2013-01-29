#!/bin/bash

#start=1017090000
#end=1017091000
start=1017000000 
end=1017001000


triggers: ISS_PDA_OUT_DQ ISS_PDB_OUT_DQ PMC_HV_MON_OUT_DQ PMC_MIXER_OUT_DQ FSS_FAST_MON_OUT_DQ FSS_MIXER_OUT_DQ FSS_PC_MON_OUT_DQ
	ligolw_add *triggers*.xml -o L_PSL_triggers.xml.gz

L1_PSL_channels.cache:
	ligo_data_find -o L -t R -s ${start} -e ${end} -u file -l > L1_PSL_channels.cache


ISS_PDA_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--initialization-file gstlal_excesspower_psl.ini  \
		--channel-name L1=PSL-ISS_PDA_OUT_DQ \
		--sample-rate 4096

ISS_PDB_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-ISS_PDB_OUT_DQ \
		--sample-rate 4096

PMC_HV_MON_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-PMC_HV_MON_OUT_DQ \
		--sample-rate 4096

PMC_MIXER_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-PMC_MIXER_OUT_DQ \
		--sample-rate 4096

FSS_FAST_MON_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-FSS_FAST_MON_OUT_D \
		--sample-rate 4096

FSS_MIXER_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-FSS_MIXER_OUT_DQ \
		--sample-rate 4096

FSS_PC_MON_OUT_DQ: L1_PSL_channels.cache
	gstlal_excesspower  \
		--verbose  \
		--data-source frames  \
		--initialization-file gstlal_excesspower_psl.ini  \
		--frame-cache L1_PSL_channels.cache \
		--gps-start-time 1017000000 \
		--gps-end-time 1017001000 \
		--channel-name L1=PSL-FSS_PC_MON_OUT_DQ \
		--sample-rate 4096

clean:
	rm -rf *.xml *.xml.gz *.cache

