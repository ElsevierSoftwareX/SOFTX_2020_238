plugin_LTLIBRARIES = libgstlal.la

libgstlal_la_SOURCES = \
	gstlal.h gstlal.c \
	gstlal_framesrc.h gstlal_framesrc.c \
	low_latency_inspiral_functions.h low_latency_inspiral_functions.c \
	gstlal_templatebank.h gstlal_templatebank.c \
	gstlal_matrixmixer.h gstlal_matrixmixer.c \
	gstlal_multiscope.h gstlal_multiscope.c \
	gstlal_simulation.h gstlal_simulation.c \
	gstlal_whiten.h gstlal_whiten.c \
	gstlal_nxydump.h gstlal_nxydump.c \
	gstadder.h gstadder.c
libgstlal_la_CFLAGS = $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS} 
libgstlal_la_LIBADD = ${gstreamer_LIBS} ${LAL_LIBS} ${GSL_LIBS} 
libgstlal_la_LDFLAGS = ${GSTREAMER_PLUGIN_LDFLAGS}

noinst_PROGRAMS = test_program
test_program_SOURCES = test_program.c \
	gstlal.h gstlal.c \
	gstlal_framesrc.h gstlal_framesrc.c \
	low_latency_inspiral_functions.h low_latency_inspiral_functions.c \
	gstlal_templatebank.h gstlal_templatebank.c \
	gstlal_matrixmixer.h gstlal_matrixmixer.c \
	gstlal_multiscope.h gstlal_multiscope.c \
	gstlal_simulation.h gstlal_simulation.c \
	gstlal_whiten.h gstlal_whiten.c \
	gstlal_nxydump.h gstlal_nxydump.c \
	gstadder.h gstadder.c
test_program_CFLAGS = -g $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS} 
test_program_LDADD = ${gstreamer_LIBS} ${LAL_LIBS} ${GSL_LIBS}

noinst_HEADERS = gstlal.h
