/*
 * IGWD frame file parser
 *
 * Copyright (C) 2012--2013  Kipp Cannon, Ed Maros
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


/*
 * ============================================================================
 *
 *                           GStreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT framecpp_igwdparse_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framecpp_igwdparse", 0, "framecpp_igwdparse element");
}


GST_BOILERPLATE_FULL(
	GstFrameCPPIGWDParse,
	framecpp_igwdparse,
	GstBaseParse,
	GST_TYPE_BASE_PARSE,
	additional_initializations
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


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


static guint16 fr_get_int_2u(GstFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_2);
}


static guint64 fr_get_int_4u(GstFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_4);
}


static guint64 fr_get_int_8u(GstFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_8);
}


static double fr_get_real_8(GstFrameCPPIGWDParse *element, GstByteReader *reader)
{
	switch(element->endianness) {
	case G_LITTLE_ENDIAN:
		return gst_byte_reader_get_float64_le_unchecked(reader);
	case G_BIG_ENDIAN:
		return gst_byte_reader_get_float64_be_unchecked(reader);
	default:
		GST_ERROR_OBJECT(element, "unrecognized endianness");
		g_assert_not_reached();
	}
}


static const gchar *fr_get_string(GstFrameCPPIGWDParse *element, GstByteReader *reader)
{
	const gchar *str;
	gst_byte_reader_skip_unchecked(reader, 2);	/* length */
	gst_byte_reader_get_string(reader, &str);
	return str;
}


static void parse_table_6(GstFrameCPPIGWDParse *element, const guint8 *data, guint64 *length, guint16 *klass)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data, element->sizeof_table_6);
	*length = fr_get_int_8u(element, &reader);
	*klass = fr_get_int_2u(element, &reader);
}


static void parse_table_7(GstFrameCPPIGWDParse *element, const guint8 *data, guint structure_length, guint16 *eof_klass, guint16 *frameh_klass)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data + element->sizeof_table_6, structure_length - element->sizeof_table_6);
	const gchar *name;

	g_assert_cmpuint(structure_length, >=, element->sizeof_table_6);	/* FIXME:  + sizeof table 7 */

	name = fr_get_string(element, &reader);
	if(!strcmp(name, FRENDOFFILE_NAME)) {
		*eof_klass = fr_get_int_2u(element, &reader);
		GST_DEBUG_OBJECT(element, "found " FRENDOFFILE_NAME " structure's class:  %hu", (unsigned short) *eof_klass);
	} else if(!strcmp(name, FRAMEH_NAME)) {
		*frameh_klass = fr_get_int_2u(element, &reader);
		GST_DEBUG_OBJECT(element, "found " FRAMEH_NAME " structure's class:  %hu", (unsigned short) *frameh_klass);
	}
}


static void parse_table_9(GstFrameCPPIGWDParse *element, const guint8 *data, guint structure_length, GstClockTime *start, GstClockTime *stop)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data + element->sizeof_table_6, structure_length - element->sizeof_table_6);
	const gchar *name;

	g_assert_cmpuint(structure_length, >=, element->sizeof_table_6);	/* FIXME:  + sizeof table 9 */

	name = fr_get_string(element, &reader);
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
	GstFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);

	/*
	 * GstBaseParse lobotomizes itself on paused-->ready transitions,
	 * so this stuff needs to be set here every time
	 */

	gst_base_parse_set_min_frame_size(parse, SIZEOF_FRHEADER);
	gst_base_parse_set_syncable(parse, FALSE);
	gst_base_parse_set_has_timing_info(parse, TRUE);

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
	GstPad *srcpad = GST_BASE_PARSE_SRC_PAD(parse);
	gboolean success = TRUE;

	caps = gst_caps_copy(gst_pad_get_pad_template_caps(srcpad));
	success = gst_pad_set_caps(srcpad, caps);
	if(!success)
		GST_ERROR_OBJECT(srcpad, "unable to set caps to %" GST_PTR_FORMAT, caps);
	gst_caps_unref(caps);

	return success;
}


/*
 * check_valid_frame()
 */


static gboolean check_valid_frame(GstBaseParse *parse, GstBaseParseFrame *frame, guint *framesize, gint *skipsize)
{
	GstFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	const guchar *data = GST_BUFFER_DATA(frame->buffer);
	gboolean file_is_complete = FALSE;

	*skipsize = 0;

	do {
		if(element->offset == 0) {
			/*
			 * parse header.  see table 5 of LIGO-T970130.
			 * note:  we only need the endianness and word
			 * sizes.
			 * FIXME:  this doesn't check that the header is
			 * valid.  adding code to do that could allow the
			 * parser to be "resyncable" (able to find frame
			 * files starting at an arbitrary point in a byte
			 * stream)
			 * FIXME:  use framecpp to parse / validate header?
			 */

			g_assert_cmpuint(GST_BUFFER_SIZE(frame->buffer), >=, SIZEOF_FRHEADER);

			/*
			 * word sizes and endianness
			 */

			element->sizeof_int_2 = *(data + 7);
			element->sizeof_int_4 = *(data + 8);
			element->sizeof_int_8 = *(data + 9);
			g_assert_cmpuint(*(data + 11), ==, 8);	/* sizeof(REAL_8) */
			if(GST_READ_UINT16_LE(data + 12) == 0x1234)
				element->endianness = G_LITTLE_ENDIAN;
			else if(GST_READ_UINT16_BE(data + 12) == 0x1234)
				element->endianness = G_BIG_ENDIAN;
			else {
				GST_ERROR_OBJECT(element, "unable to determine endianness");
				g_assert_not_reached();
			}
			GST_DEBUG_OBJECT(element, "parsed header:  endianness = %d, size of INT_2 = %d, size of INT_4 = %d, size of INT_8 = %d", element->endianness, element->sizeof_int_2, element->sizeof_int_4, element->sizeof_int_8);

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
			guint64 structure_length;
			guint16 klass;

			/*
			 * parse table 6, update file size
			 */

			parse_table_6(element, data + element->offset, &structure_length, &klass);
			*framesize = element->offset + structure_length;

			/*
			 * what to do?
			 */

			if(klass == FRSH_KLASS && (element->eof_klass == 0 || element->frameh_klass == 0)) {
				/*
				 * found frsh structure and we do not yet know the
				 * class numbers we want.  if it's complete, see if
				 * it tells us the class numbers then advance to
				 * next structure
				 */

				if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
					GST_DEBUG_OBJECT(element, "found complete %u byte FrSH structure at offset %zu", (guint) structure_length, element->offset);
					parse_table_7(element, data + element->offset, structure_length, &element->eof_klass, &element->frameh_klass);
					element->offset += structure_length;
					*framesize += element->sizeof_table_6;
				} else
					GST_DEBUG_OBJECT(element, "found incomplete %u byte FrSH structure at offset %zu, need %d more bytes", (guint) structure_length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
			} else if(klass == element->frameh_klass) {
				/*
				 * found frame header structure.  if it's complete,
				 * extract start time and duration then advance to
				 * next structure
				 */

				if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
					GstClockTime start_time, stop_time;
					GST_DEBUG_OBJECT(element, "found complete %u byte " FRAMEH_NAME " structure at offset %zu", (guint) structure_length, element->offset);
					parse_table_9(element, data + element->offset, structure_length, &start_time, &stop_time);

					element->file_start_time = MIN(element->file_start_time, start_time);
					element->file_stop_time = MAX(element->file_stop_time, stop_time);

					element->offset += structure_length;
					*framesize += element->sizeof_table_6;
				} else
					GST_DEBUG_OBJECT(element, "found incomplete %u byte " FRAMEH_NAME " structure at offset %zu, need %d more bytes", (guint) structure_length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
			} else if(klass == element->eof_klass) {
				/*
				 * found end-of-file structure.  if it's complete
				 * then the file is complete
				 */

				if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
					GST_DEBUG_OBJECT(element, "found complete %u byte " FRENDOFFILE_NAME " structure at offset %zu, have complete %u byte frame file", (guint) structure_length, element->offset, *framesize);
					element->offset = 0;
					file_is_complete = TRUE;
				} else
					GST_DEBUG_OBJECT(element, "found incomplete %u byte " FRENDOFFILE_NAME " structure at offset %zu, need %d more bytes", (guint) structure_length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
			} else {
				/*
				 * found something else.  skip to next structure
				 */

				GST_DEBUG_OBJECT(element, "found %u byte structure at offset %zu", (guint) structure_length, element->offset);
				element->offset += structure_length;
				*framesize += element->sizeof_table_6;
			}
		}
	} while(!file_is_complete && *framesize <= GST_BUFFER_SIZE(frame->buffer));

	return file_is_complete;
}


/*
 * parse_frame()
 */


static GstFlowReturn parse_frame(GstBaseParse *parse, GstBaseParseFrame *frame)
{
	GstFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	GstBuffer *buffer = frame->buffer;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * the base class sets offset and offset end to the byte offsets
	 * spanned by this file.  we are responsible for the timestamp and
	 * duration
	 */

	GST_BUFFER_TIMESTAMP(buffer) = element->file_start_time;
	GST_BUFFER_DURATION(buffer) = element->file_stop_time - element->file_start_time;

	/*
	 * mark this frame file's timestamp in the bytestream.  the start
	 * of a file is equivalent to the concept of a "key frame"
	 */

	gst_base_parse_add_index_entry(parse, GST_BUFFER_OFFSET(buffer), GST_BUFFER_TIMESTAMP(buffer), TRUE, FALSE);

	/*
	 * in practice, a collection of frame files will all be the same
	 * duration, so once we've measured the duration of one we've got a
	 * good guess for the mean frame rate
	 */

	gst_base_parse_set_frame_rate(parse, GST_SECOND, GST_BUFFER_DURATION(buffer), 0, 0);

	/*
	 * done
	 */

	GST_DEBUG_OBJECT(element, "file spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));
	return result;
}


/*
 * ============================================================================
 *
 *                             GObject Overrides
 *
 * ============================================================================
 */


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GstFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(object);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void framecpp_igwdparse_base_init(gpointer klass)
{
}


/*
 * class_init()
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"application/x-igwd-frame, " \
		"framed = (boolean) false"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"application/x-igwd-frame, " \
		"framed = (boolean) true"
	)
);


static void framecpp_igwdparse_class_init(GstFrameCPPIGWDParseClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseParseClass *parse_class = GST_BASE_PARSE_CLASS(klass);

	object_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	parse_class->start = GST_DEBUG_FUNCPTR(start);
	parse_class->set_sink_caps = GST_DEBUG_FUNCPTR(set_sink_caps);
	parse_class->check_valid_frame = GST_DEBUG_FUNCPTR(check_valid_frame);
	parse_class->parse_frame = GST_DEBUG_FUNCPTR(parse_frame);

	gst_element_class_set_details_simple(
		element_class,
		"IGWD frame file parser",
		"Codec/Parser",
		"parse byte streams into whole IGWD frame files (https://dcc.ligo.org/cgi-bin/DocDB/ShowDocument?docid=329)",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


/*
 * instance_init()
 */


static void framecpp_igwdparse_init(GstFrameCPPIGWDParse *element, GstFrameCPPIGWDParseClass *klass)
{
}
