gcc -g -c ../ssvkernel.c -I ../ -I ../../LIGOLw_xmllib `pkg-config --libs gstlal` `pkg-config --cflags gstlal`  
gcc -g -c ../background_stats_utils.c -I ../ -I ../../LIGOLw_xmllib `pkg-config --libs gstlal` `pkg-config --cflags gstlal`  
gcc -g -c test_write_stats.c `pkg-config --cflags gstlal` `pkg-config --libs gstlal` 
gcc -g -o test_write test_write_stats.o background_stats_utils.o ssvkernel.o ../../LIGOLw_xmllib/test/LIGOLwUtils.o ../../LIGOLw_xmllib/test/LIGOLwReader.o ../../LIGOLw_xmllib/test/LIGOLwWriter.o `pkg-config --cflags gstlal` `pkg-config --libs gstlal` `pkg-config --libs gsl`
