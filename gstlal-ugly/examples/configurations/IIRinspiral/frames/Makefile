# This makefile should produce a dag that when submitted will generate fake frames using
# gstlal_fake_aligo_frames --gps-start-time $(START) --gps-end-time $(END) --instrument H1 --fake-data LIGO --verbose --duration 128 --frame-type test -v
START = 869622409
END = 869722409
BREAKUP = 30000

all : dag

dag :
	./fake_frame_pipe --fake-data LIGO --duration 128 --frame-type test --gps-start-time $(START) --gps-end-time $(END) --channel-name=H1=FAKE-STRAIN --channel-name=L1=FAKE-STRAIN --max-segment-length $(BREAKUP)

# Once the dag is complete, you have to run,
# ls H1/*/*.gwf | lalapps_path2cache > H1_frame.cache
# ls H1/*/*.gwf | lalapps_path2cache > L1_frame.cache
# cat H1_frame.cache L1_frame.cache > frame.cache

realclean :
	rm -r *.sub *.dag* *.cache *.sh logs