lib_LTLIBRARIES = libgstlalspiir.la

libgstlalspiir_la_SOURCES = \
	LIGOLw_xmllib/LIGOLwReader.c \
	LIGOLw_xmllib/LIGOLwWriter.c \
	LIGOLw_xmllib/LIGOLwUtils.c


libgstlalspiir_la_CFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(XML_CFLAGS) $(gstreamer_CFLAGS) $(GSTLAL_SPIIR_CFLAGS)
libgstlalspiir_la_LIBADD = 

libgstlalspiir_la_LDFLAGS = $(AM_LDFLAGS) $(XML_LIBS) $(gstreamer_LIBS)

