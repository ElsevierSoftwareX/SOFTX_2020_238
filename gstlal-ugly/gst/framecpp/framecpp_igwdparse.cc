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
 * stuff from standard library
 */


#include <stdint.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbaseparse.h>


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


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static guint64 get_int_8u_64le(const void *data)
{
	return GST_READ_UINT64_LE(data);
}


static guint64 get_int_8u_64be(const void *data)
{
	return GST_READ_UINT64_BE(data);
}


static guint32 get_int_2u_16le(const void *data)
{
	return GST_READ_UINT16_LE(data);
}


static guint32 get_int_2u_16be(const void *data)
{
	return GST_READ_UINT16_BE(data);
}


static void parse_table_6(GSTFrameCPPIGWDParse *element, const guint8 *data, guint64 *length, guint16 *klass)
{
	*length = element->get_int_8u(data);
	*klass = element->get_int_2u(data + 8);
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
	/* number of bytes in Table 5 in LIGO-T970130 */
	const guint sizeof_frheader = 40;

	gst_base_parse_set_min_frame_size(parse, sizeof_frheader);
	gst_base_parse_set_syncable(parse, FALSE);
	gst_base_parse_set_has_timing_info(parse, FALSE);

	element->get_int_8u = NULL;
	element->get_int_2u = NULL;
	element->sizeof_table_6 = 0;
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
		GstCaps *srccaps = gst_pad_get_allowed_caps(GST_BASE_PARSE_SRC_PAD(parse));
		if(!srccaps)
			srccaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_PARSE_SRC_PAD(parse)));
		srccaps = gst_caps_make_writable(srccaps);
		gst_caps_set_simple(srccaps, "endianness", G_TYPE_INT, endianness, NULL);
		gst_pad_set_caps(GST_BASE_PARSE_SRC_PAD(parse), srccaps);

		element->endianness = endianness;
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
		 * parse header
		 */

		g_assert_cmpuint(GST_BUFFER_SIZE(frame->buffer), ==, 40);

		/*
		 * how to read INT_4 like things
		 */

		g_assert_cmpuint(*(data + 7), ==, 2);
		switch(element->endianness) {
		case G_LITTLE_ENDIAN:
			element->get_int_2u = get_int_2u_16le;
			break;
		case G_BIG_ENDIAN:
			element->get_int_2u = get_int_2u_16be;
			break;
		default:
			g_assert_not_reached();
		}

		/*
		 * how to read INT_8 like things
		 */

		g_assert_cmpuint(*(data + 9), ==, 8);
		switch(element->endianness) {
		case G_LITTLE_ENDIAN:
			element->get_int_8u = get_int_8u_64le;
			break;
		case G_BIG_ENDIAN:
			element->get_int_8u = get_int_8u_64be;
			break;
		default:
			g_assert_not_reached();
		}

		/*
		 * set the size of the structure in Table 6 of LIGO-T970130
		 */

		element->sizeof_table_6 = 8 + 2 + 4;

		/*
		 * request the next structure
		 */

		element->offset = GST_BUFFER_SIZE(frame->buffer);
		*framesize = element->offset + element->sizeof_table_6;
	} else {
		const guint16 eof_klass = 0x15; /* found in an S5 frame file */
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
			if(element->offset + length <= GST_BUFFER_SIZE(frame->buffer)) {
				element->offset = 0;
				file_is_complete = TRUE;
				GST_DEBUG_OBJECT(element, "found %u byte frame file", *framesize);
			}
		} else {
			element->offset += length;
			*framesize = element->offset + element->sizeof_table_6;
		}
	}

	return file_is_complete;
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
