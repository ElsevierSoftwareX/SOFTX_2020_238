\page gstlalmeeting20140226page Meeting on February 26, 2014

\ref gstlalmeetingspage

[TOC]

\section agenda Agenda

- lal_checktimestamps_test_01
- walk through lal_matrixmixer [ see pipeparts.mkmatrixmixer() ]

\section minutes Minutes

- lal_shift has a bug where it does not set the DISCONT flag when dynamically shifting the timestamp: Chad will fix.
- Jolien will continue to pursue an element that breaks buffer metadata
- Kipp / Chad / All to figure out a way to input buffers that would test the matrix mixer in a way that can be checked in numpy, e.g. using appsrc, or perhaps by writing a tsvdec element to decode tab separated information into buffers after being read by filesrc

\subsection actions Actions:
- Chad to fix lal_shift to properly set discont: Fixed in f9c5b20e1f2e13ad20d48da8ea83cbdf5a4d226f 

\subsection notes Notes:
