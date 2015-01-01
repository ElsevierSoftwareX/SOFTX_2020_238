EXEC = xmlRead xmlWrite spiirRead directory

CFLAGS 	= -Wall #-D__DEBUG__ 
INCLUDE = -I/usr/include/libxml2 -I/usr/local/cuda/include -I/usr/include/gstreamer-0.10
LIBS 	= -lxml2 -lz -lm -ldl
CC		= gcc

PKGCONFIG = $$(pkg-config --cflags --libs glib-2.0)

BIN	= bin
OUT = output

ALL : ${EXEC}

xmlRead : ../LIGOLwReader.c Makefile ../LIGOLwUtils.c directory
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwReader.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwUtils.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} testReader.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} testReader.o LIGOLwReader.o LIGOLwUtils.o -o ${BIN}/$@

spiirRead : ../LIGOLwReader.c Makefile ../LIGOLwUtils.c directory
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwReader.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwUtils.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} spiirReader.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} spiirReader.o LIGOLwReader.o LIGOLwUtils.o -o ${BIN}/$@

xmlWrite : ../LIGOLwWriter.c Makefile ../LIGOLwHeader.h directory
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwWriter.c -c 
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} ../LIGOLwUtils.c -c
	${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} testWriter.c -c
	# ${CC} ${CFLAGS} ${LIBS} ${INCLUDE} ${PKGCONFIG} testWriter.o LIGOLwWriter.o LIGOLwUtils.o -o ${BIN}/$@

directory : 
	@if [ ! -d ${BIN} ]; then mkdir ${BIN}; fi
	@if [ ! -d ${OUT} ]; then mkdir ${OUT}; fi

.PHONY: clean run view

clean :
	rm -f -r ${BIN}
	rm -f -r ${OUT}
	rm -f -r ./*.o

run :
	@echo run test
	# ${BIN}/xmlRead xmlIn.xml
	${BIN}/spiirRead ../../iir_H1-GSTLAL_SPLIT_BANK_0001-0-0.xml.gz
	# ${BIN}/xmlWrite ${OUT}/xmlOut.xml

view :
	vim ${OUT}/xmlOut.xml
