#!/bin/bash

echo "$0"

inputfile=data4k.bin
outputfile=output4k.bin

insamp=4096
outsamp=2048

nbchannels=1
export GST_DEBUG=5
#export GST_DEBUG_COLOR_MODE=off
#export GST_DEBUG=cuda_multiratespiir:5

#GST_DEBUG_DUMP_DOT_DIR=. 
gst-launch --gst-debug-no-color filesrc location=$inputfile ! \
  capsfilter caps=audio/x-raw-float,width=32,rate=$insamp,channels=1 !\
  cuda_multiratespiir num_depths=7 matrix=1 ! \
  capsfilter caps=audio/x-raw-float,width=32,rate=$insamp,channels=1 !\
  filesink location=$outputfile
