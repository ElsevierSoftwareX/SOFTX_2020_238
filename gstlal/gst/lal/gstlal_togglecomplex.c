/*
 * Copyright (C) 2009--2012,2014,2015 Kipp Cannon <kipp.cannon@ligo.org>
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


/**
 * SECTION:gstlal_togglecomplex
 * @short_description:  Toggle complex-valued <--> real-valued format.
 *
 * Stock GStreamer elements do not support complex-valued time series data,
 * but generally do support multi-channel time series data.  This element
 * enables the use of such elements with complex-valued time series by
 * changing the caps of complex-valued streams to make them appear to be a
 * real-valued streams with twice as many channels, and also by doing the
 * reverse.  For example, a stream containing a single channel of
 * complex-valued floating point data will be changed into two channels of
 * real-valued floating point data (the first channel is the real part, the
 * second channel the complex part).  This two-channel data can be
 * processed with, say, the stock audiofirfilter element, to apply the same
 * linear filter to both the real and imaginary components, and then the
 * stream converted back to a single channel of complex-valued data using a
 * second lal_togglecomplex element.
 *
 * This element is light-weight, it only modifies the buffer metadata.
 *
 * Reviewed:  8d9a33803cbb87f0844001a2207c5e2e55c9340c 2014-08-10 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Completed Action:
 * - Wrote unit test
 *
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
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal_togglecomplex.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define CAPS \
	"audio/x-raw, " \
	"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [2, MAX], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0; " \
	"audio/x-raw, " \
	"format = (string) {" GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


G_DEFINE_TYPE(
	GSTLALToggleComplex,
	gstlal_togglecomplex,
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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
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
			GValue x = G_VALUE_INIT;
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


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
	gboolean success = TRUE;
	guint n;

	caps = gst_caps_normalize(gst_caps_copy(caps));

	switch(direction) {
	case GST_PAD_SRC:
	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const gchar *format = gst_structure_get_string(str, "format");
			GValue channels = G_VALUE_INIT;

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32))) {
				if(g_value_is_odd_int(gst_structure_get_value(str, "channels"))) {
					GST_ERROR_OBJECT(trans, "channel count is odd");
					goto error;
				}
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z64), NULL);
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 0.5) != NULL;
			} else if(!strcmp(format, GST_AUDIO_NE(F64))) {
				if(g_value_is_odd_int(gst_structure_get_value(str, "channels"))) {
					GST_ERROR_OBJECT(trans, "channel count is odd");
					goto error;
				}
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z128), NULL);
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 0.5) != NULL;
			} else if(!strcmp(format, GST_AUDIO_NE(Z64))) {
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F32), NULL);
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 2.0) != NULL;
			} else if(!strcmp(format, GST_AUDIO_NE(Z128))) {
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
				success &= g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 2.0) != NULL;
			} else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
				goto error;
			}
			/* makes a copy of the GValue */
			gst_structure_set_value(str, "channels", &channels);
			g_value_unset(&channels);
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

	return gst_caps_simplify(caps);

error:
	gst_caps_unref(caps);
	return GST_CAPS_NONE;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * class_init()
 */


static void gstlal_togglecomplex_class_init(GSTLALToggleComplexClass *klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

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
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
}


/*
 * init()
 */


static void gstlal_togglecomplex_init(GSTLALToggleComplex *element)
{
	gst_base_transform_set_in_place(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}
