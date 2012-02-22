all : frames.cache

H1-LSC-STRAIN : 
	ligo_data_find  --observatory=H --url-type=file --type=H1_NINJA2_GAUSSIAN --gps-start-time=871147552 --gps-end-time=876357464 --lal-cache > frame.H1.cache
	#ls /home/satya/NINJA_2/NOISE/frames/noise_16k_2months/H1/*.gwf | lalapps_path2cache > frame.H1.cache

L1-LSC-STRAIN : 
	ligo_data_find  --observatory=L --url-type=file --type=L1_NINJA2_GAUSSIAN --gps-start-time=871147552 --gps-end-time=876357464 --lal-cache > frame.L1.cache
	#ls /home/satya/NINJA_2/NOISE/frames/noise_16k_2months/L1/*.gwf | lalapps_path2cache > frame.L1.cache

V1-LSC-STRAIN : 
	ligo_data_find  --observatory=V --url-type=file --type=V1_NINJA2_GAUSSIAN --gps-start-time=871147552 --gps-end-time=876357464 --lal-cache > frame.V1.cache
	#ls /home/vicere/ninja2.2months.hs/*.gwf | lalapps_path2cache > frame.V1.cache
	#sed -i 's/V NINJADATA2_HS/V V1_NINJADATA2_HS/g' frame.V1.cache # Needed because frames were not generated with conventional names

frames.cache : H1-LSC-STRAIN L1-LSC-STRAIN V1-LSC-STRAIN
	cat frame.H1.cache  frame.L1.cache frame.V1.cache > frames.cache

realclean: 
	rm -rf frame.*.cache
