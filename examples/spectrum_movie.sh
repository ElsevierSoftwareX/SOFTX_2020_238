gstlal_spectrum_movie \
	--frame-cache "/home/kipp/scratch_local/874100000-20000/cache/874100000-20000.cache" \
	--instrument "H1" \
	--channel-name "LSC-STRAIN" \
	--gps-start-time 874100000.0 \
	--gps-end-time 874120000.0 \
	--psd-fft-length 4.0 \
	--average-length 64.0 \
	--median-samples 3 \
	--output "/dev/null" \
	--verbose

exit
