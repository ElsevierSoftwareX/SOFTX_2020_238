rm -f output_H1_bank*.xml

gstlal_inspiral \
	--frame-cache "/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache" \
	--gps-start-time 874106900.0 \
	--gps-end-time 874107300.0 \
	--instrument "H1" \
	--output "output_H1.xml" \
	--template-bank=/home/kipp/Development/gstlal/examples/H1-TMPLTBANK_{02,03}-873250008-2048.xml.gz \
	--reference-psd "measured_psd.xml.gz" \
	--comment "most super-duper awsomest test" \
	--verbose

#python /home/kipp/Development/gstlal/examples/extract_channel.py 386 387 <snr_H1_bank0.dump >snr_H1_bank0.dump.new && mv -f snr_H1_bank0.dump.new snr_H1_bank0.dump
#python /home/kipp/Development/gstlal/examples/extract_channel.py 193 <chisq_H1_bank0.dump >chisq_H1_bank0.dump.new && mv -f chisq_H1_bank0.dump.new chisq_H1_bank0.dump

ligolw_sicluster --verbose --cluster-window 10.0 output_H1.xml
python /home/kipp/Development/gstlal/examples/dump_triggers.py output_H1.xml >triggers.dump

exit

	--injections /home/kipp/Development/gstlal/examples/HL-INJECTIONS_1_BNS_INJ-873247860-176894.xml \
	--template-bank=/home/kipp/Development/gstlal/examples/H1-TMPLTBANK_{00,01,02,03,04,05,06,07,08,09}-873250008-2048.xml.gz \

	# output for hardware injection @ 874107078.149271066
	--nxydump-segment 874107058.0:874107098.0 \

	# output for impulse injection @ 873337860
	--nxydump-segment 873337850.0:873337960.0 \

	# output for use with software injections:
	# bns_injections.xml = 874107198.405080859, impulse =
	# 874107189
	--nxydump-segment 874107188.0:874107258.0 \

	# FIXME:  what's at this time?
	--nxydump-segment 873248760.0:873248960.0 \

	# output to dump lots and lots of data (the whole cache)
	--nxydump-segment 873247860.0:873424754.0 \
