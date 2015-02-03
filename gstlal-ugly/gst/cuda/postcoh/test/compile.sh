
gcc -g -c test_detmap.c `pkg-config --cflags gstlal` `pkg-config --libs gstlal` `pkg-config --cflags chealpix` `pkg-config --libs chealpix` -llapack -llapacke
gcc -g -o test_detmap test_detmap.o ../../LIGOLw_xmllib/test/LIGOLwUtils.o ../../LIGOLw_xmllib/test/LIGOLwWriter.o `pkg-config --cflags gstlal` `pkg-config --libs gstlal` `pkg-config --cflags chealpix` `pkg-config --libs chealpix` -llapack -llapacke
