/*
 * IGWD frame file parser
 *
 * Copyright (C) 2012  Kipp Cannon, Ed Maros
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ========================================================================
 *
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbaseparse.h>
#include <gst/base/gstbytereader.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <framecpp_igwdparse.h>


GST_DEBUG_CATEGORY(framecpp_igwdparse_debug);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT framecpp_igwdparse_debug

/* FIXME:  get from framecpp */
/* number of bytes in table 5 in LIGO-T970130 */
#define SIZEOF_FRHEADER 40
/* class of FrSH structure */
#define FRSH_KLASS 1
/* name of end-of-file structure */
#define FRENDOFFILE_NAME "FrEndOfFile"
/* name of frame header structure */
#define FRAMEH_NAME "FrameH"


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/* FIXME:  get from framecpp */

static guint64 fr_get_int_u(GstByteReader *reader, gint endianness, gint size)
{
	switch(endianness) {
	case G_LITTLE_ENDIAN:
		switch(size) {
		case 2: return gst_byte_reader_get_uint16_le_unchecked(reader);
		case 3: return gst_byte_reader_get_uint24_le_unchecked(reader);
		case 4: return gst_byte_reader_get_uint32_le_unchecked(reader);
		case 8: return gst_byte_reader_get_uint64_le_unchecked(reader);
		default:
			GST_ERROR("unsupported word size");
			g_assert_not_reached();
		}
	case G_BIG_ENDIAN:
		switch(size) {
		case 2: return gst_byte_reader_get_uint16_be_unchecked(reader);
		case 3: return gst_byte_reader_get_uint24_be_unchecked(reader);
		case 4: return gst_byte_reader_get_uint32_be_unchecked(reader);
		case 8: return gst_byte_reader_get_uint64_be_unchecked(reader);
		default:
			GST_ERROR("unsupported word size");
			g_assert_not_reached();
		}
	default:
		GST_ERROR("unrecognized endianness");
		g_assert_not_reached();
	}
}


static guint16 fr_get_int_2u(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_2);
}


static guint64 fr_get_int_4u(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_4);
}


static guint64 fr_get_int_8u(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_8);
}


static double fr_get_real_8(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	switch(element->endianness) {
	case G_LITTLE_ENDIAN:
		return gst_byte_reader_get_float64_le_unchecked(reader);
	case G_BIG_ENDIAN:
		return gst_byte_reader_get_float64_be_unchecked(reader);
	default:
		GST_ERROR("unrecognized endianness");
		g_assert_not_reached();
	}
}


static const gchar *fr_get_string(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	const gchar *str;
	gst_byte_reader_skip_unchecked(reader, 2);	/* length */
	gst_byte_reader_get_string(reader, &str);
	return str;
}


static void parse_table_6(GSTFrameCPPIGWDParse *element, const guint8 *data, guint64 *length, guint16 *klass)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data, element->sizeof_table_6);
	*length = fr_get_int_8u(element, &reader);
	*klass = fr_get_int_2u(element, &reader);
}


static void parse_table_7(GSTFrameCPPIGWDParse *element, const guint8 *data, guint length, guint16 *eof_klass, guint16 *frameh_klass)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data + element->sizeof_table_6, length - element->sizeof_table_6);
	const gchar *name = fr_get_string(element, &reader);

	if(!strcmp(name, FRENDOFFILE_NAME)) {
		*eof_klass = fr_get_int_2u(element, &reader);
		GST_DEBUG_OBJECT(element, "found " FRENDOFFILE_NAME " structure's class:  %d", (int) *eof_klass);
	} else if(!strcmp(name, FRAMEH_NAME)) {
		*frameh_klass = fr_get_int_2u(element, &reader);
		GST_DEBUG_OBJECT(element, "found " FRAMEH_NAME " structure's class:  %d", (int) *frameh_klass);
	}
}


static void parse_table_9(GSTFrameCPPIGWDParse *element, const guint8 *data, guint length, GstClockTime *start, GstClockTime *stop)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data + element->sizeof_table_6, length - element->sizeof_table_6);
	const gchar *name = fr_get_string(element, &reader);

	gst_byte_reader_skip_unchecked(&reader, 3 * element->sizeof_int_4);	/* run, frame, dataQuality */
	*start = fr_get_int_4u(element, &reader) * GST_SECOND + fr_get_int_4u(element, &reader);
	gst_byte_reader_skip_unchecked(&reader, element->sizeof_int_2);	/* ULeapS */
	*stop = *start + (GstClockTime) round(fr_get_real_8(element, &reader) * GST_SECOND);
	GST_DEBUG_OBJECT(element, "found " FRAMEH_NAME " \"%s\" spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", name, GST_TIME_SECONDS_ARGS(*start), GST_TIME_SECONDS_ARGS(*stop));
}


/*
 * ============================================================================
 *
 *                           GstBaseParse Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */


static gboolean start(GstBaseParse *parse)
{
	GSTFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);

	/*
	 * GstBaseParse lobotomizes itself on paused-->ready transitions,
	 * so this stuff needs to be set here every time
	 */

	gst_base_parse_set_min_frame_size(parse, SIZEOF_FRHEADER);
	gst_base_parse_set_syncable(parse, FALSE);
	gst_base_parse_set_has_timing_info(parse, FALSE);	/* FIXME:  should be TRUE, but breaks element.  why? */

	/*
	 * everything else will be reset when the header is parsed
	 */

	element->offset = 0;

	return TRUE;
}


/*
 * set_sink_caps()
 */


static gboolean set_sink_caps(GstBaseParse *parse, GstCaps *caps)
{
	GSTFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	GstStructure *s;
	gint endianness;
	gboolean success = TRUE;

	s = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(s, "endianness", &endianness)) {
		GST_DEBUG_OBJECT(element, "unable to parse endianness from %" GST_PTR_FORMAT, caps);
		success = FALSE;
	}

	if(success) {
		element->endianness = endianness;
		GST_DEBUG_OBJECT(element, "endianness set to %d", endianness);

		caps = gst_caps_copy(caps);
		gst_caps_set_simple(caps, "framed", G_TYPE_BOOLEAN, TRUE, NULL);
		gst_pad_set_caps(GST_BASE_PARSE_SRC_PAD(parse), caps);
		gst_caps_unref(caps);
	}

	return success;
}


/*
 * check_valid_frame()
 */


static gboolean check_valid_frame(GstBaseParse *parse, GstBaseParseFrame *frame, guint *framesize, gint *skipsize)
{
	GSTFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	const guchar *data = GST_BUFFER_DATA(frame->buffer);
	gboolean file_is_complete = FALSE;

	*skipsize = 0;

	if(element->offset == 0) {
		/*
		 * parse header.  see table 5 of LIGO-T970130.  note:  we
		 * know the endianness from the caps (which the typefinder
		 * takes care of figuring out for us), so the only other
		 * things we need to figure out are the word sizes.
		 * FIXME:  get from framecpp?
		 */

		GST_DEBUG_OBJECT(element, "parsing file header");
		g_assert_cmpuint(GST_BUFFER_SIZE(frame->buffer), >=, SIZEOF_FRHEADER);

		/*
		 * word sizes
		 */

		element->sizeof_int_2 = *(data + 7);
		element->sizeof_int_4 = *(data + 8);
		element->sizeof_int_8 = *(data + 9);
		g_assert_cmpuint(*(data + 11), ==, 8);	/* sizeof(REAL_8) */
		GST_DEBUG_OBJECT(element, "endianness = %d, size of INT_2 = %d, size of INT_4 = %d, size of INT_8 = %d", element->endianness, element->sizeof_int_2, element->sizeof_int_4, element->sizeof_int_8);

		/*
		 * set the size of the structure in table 6 of LIGO-T970130
		 */

		element->sizeof_table_6 = element->sizeof_int_8 + element->sizeof_int_2 + element->sizeof_int_4;

		/*
		 * reset the class numbers
		 */

		element->eof_klass = 0;
		element->frameh_klass = 0;

		/*
		 * reset the start and stop times
		 */

		element->file_start_time = -1;	/* max GstClockTime */
		element->file_stop_time = 0;	/* min GstClockTime */

		/*
		 * request the first table 6 structure
		 */

		element->offset = SIZEOF_FRHEADER;
		*framesize = element->offset + element->sizeof_table_6;
	} else {
		guint64 length;
		guint16 klass;

		/*
		 * parse table 6, update file size
		 */

		parse_table_6(element, data + element->offset, &length, &klass);
		*framesize = element->offset + length;

		/*
		 * what to do?
		 */

		if(klass == FRSH_KLASS && (element->eof_klass == 0 || element->frameh_klass == 0)) {
			/*
			 * found frsh structure and do we not yet know the
			 * class numbers we want.  if it's complete, see if
			 * it tells us the class numbers then advance to
			 * next structure
			 */

			if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
				GST_DEBUG_OBJECT(element, "found complete %u byte FrSH structure at offset %zu", (guint) length, element->offset);
				parse_table_7(element, data + element->offset, length, &element->eof_klass, &element->frameh_klass);
				element->offset += length;
				*framesize += element->sizeof_table_6;
			} else
				GST_DEBUG_OBJECT(element, "found incomplete %u byte FrSH structure at offset %zu, need %d more bytes", (guint) length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
		} else if(klass == element->frameh_klass) {
			/*
			 * found frame header structure.  if it's complete,
			 * extract start time and duration then advance to
			 * next structure
			 */

			if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
				GstClockTime start_time, stop_time;
				GST_DEBUG_OBJECT(element, "found complete %u byte " FRAMEH_NAME " structure at offset %zu", (guint) length, element->offset);
				parse_table_9(element, data + element->offset, length, &start_time, &stop_time);

				element->file_start_time = MIN(element->file_start_time, start_time);
				element->file_stop_time = MAX(element->file_stop_time, stop_time);

				element->offset += length;
				*framesize += element->sizeof_table_6;
			} else
				GST_DEBUG_OBJECT(element, "found incomplete %u byte " FRAMEH_NAME " structure at offset %zu, need %d more bytes", (guint) length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
		} else if(klass == element->eof_klass) {
			/*
			 * found end-of-file structure.  if it's complete
			 * then the file is complete
			 */

			if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
				GST_DEBUG_OBJECT(element, "found complete %u byte " FRENDOFFILE_NAME " structure at offset %zu, have complete %u byte frame file", (guint) length, element->offset, *framesize);
				element->offset = 0;
				file_is_complete = TRUE;
			} else
				GST_DEBUG_OBJECT(element, "found incomplete %u byte " FRENDOFFILE_NAME " structure at offset %zu, need %d more bytes", (guint) length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
		} else {
			/*
			 * found something else.  skip to next structure
			 */

			GST_DEBUG_OBJECT(element, "found %u byte structure at offset %zu", (guint) length, element->offset);
			element->offset += length;
			*framesize += element->sizeof_table_6;
		}
	}

	return file_is_complete;
}


/*
 * parse_frame()
 */


static GstFlowReturn parse_frame(GstBaseParse *parse, GstBaseParseFrame *frame)
{
	GSTFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	GstBuffer *buffer = frame->buffer;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * offset and offset end are automatically set to the byte offsets
	 * spanned by this file
	 */

	GST_BUFFER_TIMESTAMP(buffer) = element->file_start_time;
	GST_BUFFER_DURATION(buffer) = element->file_stop_time - element->file_start_time;

	GST_DEBUG_OBJECT(element, "file spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));
	return result;
}


/*
 * ============================================================================
 *
 *                           GStreamer Element
 *
 * ============================================================================
 */


/*
 * pad templates
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"application/x-igwd-frame, " \
		"endianness = (int) {1234, 4321}, " \
		"framed = (boolean) false"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"application/x-igwd-frame, " \
		"endianness = (int) {1234, 4321}, " \
		"framed = (boolean) true"
	)
);


/*
 * base_init()
 */


static void framecpp_igwdparse_base_init(gpointer klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseParseClass *parse_class = GST_BASE_PARSE_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"IGWD frame file parser",
		"Codec/Parser",
		"Parse byte streams into whole IGWD frame files",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	parse_class->start = GST_DEBUG_FUNCPTR(start);
	parse_class->set_sink_caps = GST_DEBUG_FUNCPTR(set_sink_caps);
	parse_class->check_valid_frame = GST_DEBUG_FUNCPTR(check_valid_frame);
	parse_class->parse_frame = GST_DEBUG_FUNCPTR(parse_frame);
}


/*
 * class_init()
 */


static void framecpp_igwdparse_class_init(GSTFrameCPPIGWDParseClass *klass)
{
}


/*
 * instance_init()
 */


static void framecpp_igwdparse_init(GSTFrameCPPIGWDParse *object, GSTFrameCPPIGWDParseClass *klass)
{
}


/*
 * boilerplate
 */


GST_BOILERPLATE(
	GSTFrameCPPIGWDParse,
	framecpp_igwdparse,
	GstBaseParse,
	GST_TYPE_BASE_PARSE
);
