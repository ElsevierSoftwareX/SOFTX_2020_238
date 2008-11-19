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
	gstlalcollectpads.h gstlalcollectpads.c \
	gstadder.h gstadder.c \
	gstlal_triggergen.h gstlal_triggergen.c \
	gstlal_gate.h gstlal_gate.c \
	gstlal_chisquare.h gstlal_chisquare.c
libgstlal_la_CFLAGS = -O2 $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS} 
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
	gstlalcollectpads.h gstlalcollectpads.c \
	gstadder.h gstadder.c \
	gstlal_triggergen.h gstlal_triggergen.c \
	gstlal_gate.h gstlal_gate.c \
	gstlal_chisquare.h gstlal_chisquare.c
test_program_CFLAGS = -g -O2 $(AM_CFLAGS) $(CFLAGS) ${gstreamer_CFLAGS} ${LAL_CFLAGS} ${GSL_CFLAGS} 
test_program_LDADD = ${gstreamer_LIBS} ${LAL_LIBS} ${GSL_LIBS}

noinst_HEADERS = gstlal.h
