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


G_DEFINE_TYPE_WITH_CODE(
	GstFrameCPPIGWDParse,
	framecpp_igwdparse,
	GST_TYPE_BASE_PARSE,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framecpp_igwdparse", 0, "framecpp_igwdparse element")
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
	 *
	 * FIXME:  see if this can now be moved to init()
	 */

	gst_base_parse_set_min_frame_size(parse, SIZEOF_FRHEADER);
	gst_base_parse_set_syncable(parse, FALSE);
	gst_base_parse_set_has_timing_info(parse, TRUE);

	/*
	 * everything else will be reset when the header is parsed
	 */

	element->filesize = SIZEOF_FRHEADER;
	element->offset = 0;

	return TRUE;
}


/*
 * set_sink_caps()
 */


static gboolean set_sink_caps(GstBaseParse *parse, GstCaps *caps)
{
	GstPad *srcpad = GST_BASE_PARSE_SRC_PAD(parse);
	GstStructure *s;
	gboolean framed;
	gboolean success = TRUE;

	/*
	 * pass-through if input is already framed
	 */

	s = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_boolean(s, "framed", &framed);

	if(success) {
		gst_pad_push_event(srcpad, gst_event_new_caps(gst_pad_get_pad_template_caps(srcpad)));
		gst_base_parse_set_passthrough(parse, framed);
	} else
		GST_ERROR_OBJECT(parse, "unable to accept sink caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * handle_frame()
 */


static GstFlowReturn handle_frame(GstBaseParse *parse, GstBaseParseFrame *frame, gint *skipsize)
{
	GstFrameCPPIGWDParse *element = FRAMECPP_IGWDPARSE(parse);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	if(!gst_buffer_map(frame->buffer, &mapinfo, GST_MAP_READ)) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("buffer cannot be mapped for read"));
		return GST_FLOW_ERROR;
	}

	*skipsize = 0;

	while(element->filesize <= mapinfo.size) {
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

			g_assert_cmpuint(mapinfo.size, >=, SIZEOF_FRHEADER);

			/*
			 * word sizes and endianness
			 */

			element->sizeof_int_2 = *(mapinfo.data + 7);
			element->sizeof_int_4 = *(mapinfo.data + 8);
			element->sizeof_int_8 = *(mapinfo.data + 9);
			g_assert_cmpuint(*(mapinfo.data + 11), ==, 8);	/* sizeof(REAL_8) */
			if(GST_READ_UINT16_LE(mapinfo.data + 12) == 0x1234)
				element->endianness = G_LITTLE_ENDIAN;
			else if(GST_READ_UINT16_BE(mapinfo.data + 12) == 0x1234)
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
			 * reset the class numbers to impossible values
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
			element->filesize = element->offset + element->sizeof_table_6;
		} else {
			/* FIXME:  add checksumming to allow resync on incomplete/partial files */

			guint64 structure_length;
			guint16 klass;

			/*
			 * parse table 6, update file size
			 */

			g_assert_cmpuint(mapinfo.size, >=, element->offset + element->sizeof_table_6);
			parse_table_6(element, mapinfo.data + element->offset, &structure_length, &klass);
			element->filesize = element->offset + structure_length;

			if(element->filesize > mapinfo.size) {
				GST_DEBUG_OBJECT(element, "need more data to complete %u byte structure at offset %zu", (guint) structure_length, element->offset);
				break;
			}

			/*
			 * what to do?
			 */

			if(klass == FRSH_KLASS && (element->eof_klass == 0 || element->frameh_klass == 0)) {
				/*
				 * found frsh structure and we do not yet
				 * know the class numbers we want.  see if
				 * it tells us the class numbers
				 */

				GST_DEBUG_OBJECT(element, "found %u byte FrSH structure at offset %zu", (guint) structure_length, element->offset);
				parse_table_7(element, mapinfo.data + element->offset, structure_length, &element->eof_klass, &element->frameh_klass);
			} else if(klass == element->frameh_klass) {
				/*
				 * found frame header structure.  extract
				 * start time and duration
				 */

				GstClockTime start_time, stop_time;
				GST_DEBUG_OBJECT(element, "found %u byte " FRAMEH_NAME " structure at offset %zu", (guint) structure_length, element->offset);
				parse_table_9(element, mapinfo.data + element->offset, structure_length, &start_time, &stop_time);

				element->file_start_time = MIN(element->file_start_time, start_time);
				element->file_stop_time = MAX(element->file_stop_time, stop_time);
			} else if(klass == element->eof_klass) {
				/*
				 * found end-of-file structure.  the file
				 * is complete
				 */

				GST_DEBUG_OBJECT(element, "found %u byte " FRENDOFFILE_NAME " structure at offset %zu, have complete %zu byte frame file", (guint) structure_length, element->offset, element->filesize);

				/*
				 * the base class sets offset and offset
				 * end to the byte offsets spanned by this
				 * file.  we are responsible for the
				 * timestamp and duration
				 */

				GST_BUFFER_TIMESTAMP(frame->buffer) = element->file_start_time;
				GST_BUFFER_DURATION(frame->buffer) = element->file_stop_time - element->file_start_time;
				GST_DEBUG_OBJECT(element, "file spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(frame->buffer));

				/*
				 * mark this frame file's timestamp in the
				 * bytestream.  the start of a file is
				 * equivalent to the concept of a "key
				 * frame" as it is only possible to seek to
				 * file boundaries
				 */

				gst_base_parse_add_index_entry(parse, GST_BUFFER_OFFSET(frame->buffer), GST_BUFFER_TIMESTAMP(frame->buffer), TRUE, FALSE);

				/*
				 * in practice, a collection of frame files
				 * will all be the same duration, so once
				 * we've measured the duration of one we've
				 * got a good guess for the mean frame rate
				 */

				gst_base_parse_set_frame_rate(parse, GST_SECOND, GST_BUFFER_DURATION(frame->buffer), 0, 0);

				/*
				 * done.  need to unmap buffer before
				 * calling _finish_frame()
				 */

				gst_buffer_unmap(frame->buffer, &mapinfo);
				result = gst_base_parse_finish_frame(parse, frame, element->filesize);

				/*
				 * reset for next file, and exit loop
				 */

				element->filesize = SIZEOF_FRHEADER;
				element->offset = 0;
				return result;
			} else {
				/*
				 * found something else
				 */

				GST_DEBUG_OBJECT(element, "found uninteresting %u byte structure at offset %zu", (guint) structure_length, element->offset);
			}

			/*
			 * advance to next structure
			 */

			element->offset = element->filesize;
			element->filesize += element->sizeof_table_6;
		}
	}

	gst_buffer_unmap(frame->buffer, &mapinfo);

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

	(void) element;	/* silence unused variable warning */

	G_OBJECT_CLASS(framecpp_igwdparse_parent_class)->finalize(object);
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
		"framed = (boolean) {true, false}"
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
	parse_class->handle_frame = GST_DEBUG_FUNCPTR(handle_frame);

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
 *
 * see also start() override
 */


static void framecpp_igwdparse_init(GstFrameCPPIGWDParse *element)
{
}
