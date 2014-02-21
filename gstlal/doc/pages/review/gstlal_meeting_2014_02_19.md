\page gstlalmeeting20140219page Meeting on February 19, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- walk through gstlal_matrixmixer

\section minutes Minutes

- walked through gstlal_matrixmixer.h header file and static functions gstlal_matrixmixer_class_init and transform in gstlal_matrixmixer.c

Actions
- Jolien to write a unit test code for lal_checktimestamps
- Chad has taken a stab at something that might help and checked it into gstlal/gstlal/tests.  This test program dynamically adds a one second time shift every time the user his ctrl+C.  You need to do kill -9 to stop the program ;)

		$ ./lal_checktimestamps_test_01.py 
		src (00:00:05): 5 seconds
		^Cshifting by 1 ns
		lal_checktimestamps+lal_checktimestamps0: got timestamp 7.000666617 s expected 7.000666616 s (discont flag is not set)
		^Cshifting by 2 ns
		src (00:00:10): 10 seconds
		lal_checktimestamps+lal_checktimestamps0: got timestamp 10.000666618 s expected 10.000666617 s (discont flag is not set)
		lal_checktimestamps+lal_checktimestamps0: timestamp/offset mismatch:  got timestamp 10.000666618 s, buffer offset 20480 corresponds to timestamp 10.000666616 s (error = 2 ns)
		lal_checktimestamps+lal_checktimestamps0: timestamp/offset mismatch:  got timestamp 11.000666618 s, buffer offset 22528 corresponds to timestamp 11.000666616 s (error = 2 ns)	

Notes:
- N/A
