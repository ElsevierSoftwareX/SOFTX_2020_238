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
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbaseparse.h>
#include <gst/base/gstbytereader.h>


/*
 * our own stuff
 */


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

/* number of bytes in table 5 in LIGO-T970130 */
/* FIXME:  get from framecpp */
#define SIZEOF_FRHEADER 40


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
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_2u);
}


static guint64 fr_get_int_8u(GSTFrameCPPIGWDParse *element, GstByteReader *reader)
{
	return fr_get_int_u(reader, element->endianness, element->sizeof_int_8u);
}


static void parse_table_6(GSTFrameCPPIGWDParse *element, const guint8 *data, guint64 *length, guint16 *klass)
{
	GstByteReader reader = GST_BYTE_READER_INIT(data, element->sizeof_table_6);
	*length = fr_get_int_8u(element, &reader);
	*klass = fr_get_int_2u(element, &reader);
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

	gst_base_parse_set_min_frame_size(parse, SIZEOF_FRHEADER);
	gst_base_parse_set_syncable(parse, FALSE);
	gst_base_parse_set_has_timing_info(parse, FALSE);

	element->sizeof_int_2u = 0;
	element->sizeof_int_8u = 0;
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
		gint sizeof_int_4u;

		/*
		 * parse header.  see table 5 of LIGO-T970130.  note:  we
		 * know the endianness from the caps (which the typefinder
		 * takes care of figuring out for us), so the only other
		 * things we need to figure out are the word sizes.
		 * FIXME:  get from framecpp?
		 */

		g_assert_cmpuint(GST_BUFFER_SIZE(frame->buffer), >=, SIZEOF_FRHEADER);

		/*
		 * word sizes
		 */

		element->sizeof_int_2u = *(data + 7);
		sizeof_int_4u = *(data + 8);
		element->sizeof_int_8u = *(data + 9);
		GST_DEBUG_OBJECT(element, "endianness = %d, size of INT_2 = %d, size of INT_4 = %d, size of INT_8 = %d", element->endianness, element->sizeof_int_2u, sizeof_int_4u, element->sizeof_int_8u);

		/*
		 * set the size of the structure in table 6 of LIGO-T970130
		 */

		element->sizeof_table_6 = element->sizeof_int_8u + element->sizeof_int_2u + sizeof_int_4u;

		/*
		 * request the first table 6 structure
		 */

		element->offset = SIZEOF_FRHEADER;
		*framesize = element->offset + element->sizeof_table_6;
	} else {
		/* end-of-file class number.  found in an S5 frame file. */
		/* FIXME:  get from framecpp */
		const guint16 eof_klass = 0x15;
		guint64 length;
		guint16 klass;

		/*
		 * parse table 6
		 */

		parse_table_6(element, data + element->offset, &length, &klass);

		/*
		 * frame file complete?  if not request more data
		 */

		if(klass == eof_klass) {
			*framesize = element->offset + length;
			if(*framesize <= GST_BUFFER_SIZE(frame->buffer)) {
				GST_DEBUG_OBJECT(element, "found %u byte FrEndOfFile structure at offset %zu, have complete %u byte frame file", (guint) length, element->offset, *framesize);
				element->offset = 0;
				file_is_complete = TRUE;
			} else
				GST_DEBUG_OBJECT(element, "found %u byte FrEndOfFile structure at offset %zu, need %d more bytes", (guint) length, element->offset, *framesize - GST_BUFFER_SIZE(frame->buffer));
		} else {
			GST_DEBUG_OBJECT(element, "found %u byte non-FrEndOfFile structure at offset %zu", (guint) length, element->offset);
			element->offset += length;
			*framesize = element->offset + element->sizeof_table_6;
		}
	}

	return file_is_complete;
}


/*
 * parse_frame()
 */


static GstFlowReturn parse_frame(GstBaseParse *parse, GstBaseParseFrame *frame)
{
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * FIXME:  set timestamp, duration, offset of frame->buffer
	 */

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
