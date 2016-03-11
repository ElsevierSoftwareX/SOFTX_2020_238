
gcc -g -c gen_detrsp_maps.c `pkg-config --cflags gstlal` `pkg-config --libs gstlal` `pkg-config --cflags chealpix` `pkg-config --libs chealpix` -llapack -llapacke
gcc -g -o gen_detrsp_maps gen_detrsp_maps.o ../../LIGOLw_xmllib/test/LIGOLwUtils.o ../../LIGOLw_xmllib/test/LIGOLwWriter.o `pkg-config --cflags gstlal` `pkg-config --libs gstlal` `pkg-config --cflags chealpix` `pkg-config --libs chealpix` -llapack -llapacke
