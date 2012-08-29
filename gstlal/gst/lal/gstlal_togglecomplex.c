/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 *  stuff from the C library
 */


#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal_togglecomplex.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [2, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [2, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


GST_BOILERPLATE(
	GSTLALToggleComplex,
	gstlal_togglecomplex,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint channels, width;
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);
	success &= gst_structure_get_int(str, "width", &width);

	if(success)
		*size = width / 8 * channels;
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static gint scale_int(gint x, double factor, gint min, gint max)
{
	if(factor >= 1)
		return x < max / factor ? x * factor : max;
	return x > min / factor ? x * factor : min;
}


static gboolean g_value_is_odd_int(const GValue *x)
{
	return G_VALUE_HOLDS_INT(x) && (g_value_get_int(x) & 1);
}


static GValue *g_value_scale_int(const GValue *src, GValue *dst, double factor)
{
	if(G_VALUE_HOLDS_INT(src)) {
		g_value_init(dst, G_TYPE_INT);
		g_value_set_int(dst, scale_int(g_value_get_int(src), factor, 1, G_MAXINT));
	} else if(GST_VALUE_HOLDS_INT_RANGE(src)) {
		g_value_init(dst, GST_TYPE_INT_RANGE);
		gst_value_set_int_range(dst, scale_int(gst_value_get_int_range_min(src), factor, 1, G_MAXINT), scale_int(gst_value_get_int_range_max(src), factor, 1, G_MAXINT));
	} else if(GST_VALUE_HOLDS_LIST(src)) {
		guint i;
		g_value_init(dst, GST_TYPE_LIST);
		for(i = 0; i < gst_value_list_get_size(src); i++) {
			GValue x = {0};
			if(!g_value_scale_int(gst_value_list_get_value(src, i), &x, factor)) {
				g_value_unset(dst);
				return NULL;
			}
			/* makes a copy of the GValue */
			gst_value_list_append_value(dst, &x);
			g_value_unset(&x);
		}
	} else {
		g_assert_not_reached();
		return NULL;
	}
	return dst;
}


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	gboolean success = TRUE;
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const gchar *name = gst_structure_get_name(str);
			GValue channels = {0};
			GValue width = {0};
			if(name && !strcmp(name, "audio/x-raw-float")) {
				gst_structure_set_name(str, "audio/x-raw-complex");
				if(g_value_is_odd_int(gst_structure_get_value(str, "channels"))) {
					GST_ERROR_OBJECT(trans, "channel count is odd");
					goto error;
				}
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 0.5) != NULL;
				success &= g_value_scale_int(gst_structure_get_value(str, "width"), &width, 2.0) != NULL;
			} else if(name && !strcmp(name, "audio/x-raw-complex")) {
				gst_structure_set_name(str, "audio/x-raw-float");
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 2.0) != NULL;
				success &= g_value_scale_int(gst_structure_get_value(str, "width"), &width, 0.5) != NULL;
			} else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, name ? name : "(NULL)", caps);
				goto error;
			}
			/* makes a copy of the GValue */
			gst_structure_set_value(str, "channels", &channels);
			gst_structure_set_value(str, "width", &width);
			g_value_unset(&channels);
			g_value_unset(&width);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	}

	if(!success) {
		GST_ERROR_OBJECT(trans, "failure constructing caps");
		goto error;
	}

	return caps;

error:
	gst_caps_unref(caps);
	return GST_CAPS_NONE;
}


/*
 * prepare_output_buffer()
 *
 * FIXME:  the logic here results in a copy being made of the buffer's
 * metadata even if this element is the only element with a reference to
 * the input buffer.  it migh be possible to avoid this in 0.11
 */


static GstFlowReturn prepare_output_buffer(GstBaseTransform *trans, GstBuffer *input, gint size, GstCaps *caps, GstBuffer **buf)
{
	/*
	 * start by making output a reference to the input
	 */

	gst_buffer_ref(input);
	*buf = input;

	/*
	 * make metadata writeable
	 */

	*buf = gst_buffer_make_metadata_writable(*buf);
	if(!*buf) {
		GST_DEBUG_OBJECT(trans, "failure creating sub-buffer from input");
		return GST_FLOW_ERROR;
	}

	/*
	 * replace caps with output caps
	 */

	gst_buffer_set_caps(*buf, caps);

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * base_init()
 */


static void gstlal_togglecomplex_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Toggle Complex",
		"Filter/Audio",
		"Replace float caps with complex (with half the channels), complex with float (with twice the channels).",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
}


/*
 * class_init()
 */


static void gstlal_togglecomplex_class_init(GSTLALToggleComplexClass *klass)
{
}


/*
 * init()
 */


static void gstlal_togglecomplex_init(GSTLALToggleComplex *element, GSTLALToggleComplexClass *klass)
{
	GST_BASE_TRANSFORM(element)->always_in_place = TRUE;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}
