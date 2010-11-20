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
 * stuff from C
 */


#include <math.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_sumsquares.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


#define DEFINE_FUNC(T, func) \
static GstFlowReturn sumsquares_ ## T (GSTLALSumSquares *element, GstBuffer *inbuf, GstBuffer *outbuf) \
{ \
	const T *src = (const T *) GST_BUFFER_DATA(inbuf); \
	T *dst = (T *) GST_BUFFER_DATA(outbuf); \
	T *dst_end = dst + (GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf)); \
	const T *weights = element->weights_ ## T; \
\
	if(weights) { \
		for(; dst < dst_end; dst++) { \
			const T *src_end = src + element->channels; \
			const T *w = weights; \
			for(*dst = 0; src < src_end; w++, src++) \
				*dst += func(*w * *src, 2); \
		} \
	} else { \
		for(; dst < dst_end; dst++) { \
			const T *src_end = src + element->channels; \
			for(*dst = 0; src < src_end; src++) \
				*dst += func(*src, 2); \
		} \
	} \
\
	return GST_FLOW_OK; \
}


DEFINE_FUNC(double, pow)
DEFINE_FUNC(float, powf)


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
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}"
	)
);


GST_BOILERPLATE(
	GSTLALSumSquares,
	gstlal_sumsquares,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_WEIGHTS = 1
};


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
	gint channels;
	gint width;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = width / 8 * channels;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it can have any number of channels or, if the length of
		 * the weights vector is known, the number of channels must
		 * equal the number of weights
		 */

		g_mutex_lock(element->weights_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->weights_double)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, element->channels, NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(element->weights_lock);
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it must have only 1 channel
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, 1, NULL);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(trans);
	GstStructure *s;
	gint channels, width;

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(s, "width", &width)) {
		GST_DEBUG_OBJECT(element, "unable to parse width from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	if(!element->weights_double)
		element->channels = channels;
	else if(channels != element->channels) {
		/* FIXME:  perhaps emit a "channel-count-changed" signal? */
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, element->channels, incaps);
		return FALSE;
	}

	element->is_float = (width == 32);

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(trans);
	GstFlowReturn result;

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		g_mutex_lock(element->weights_lock);
		if(element->is_float) {
			if(element->weights_double && !element->weights_float) {
				gint i;
				element->weights_float = g_malloc(element->channels * sizeof(float));
				for(i = 0; i < element->channels; i ++)
					element->weights_float[i] = element->weights_double[i];
			}
			result = sumsquares_float(element, inbuf, outbuf);
		} else {
			result = sumsquares_double(element, inbuf, outbuf);
		}
		g_mutex_unlock(element->weights_lock);
	} else {
		/*
		 * input is 0s.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		result = GST_FLOW_OK;
	}

	/*
	 * done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_WEIGHTS: {
		gint channels;
		g_mutex_lock(element->weights_lock);
		if(element->weights_double) {
			channels = element->channels;
			g_free(element->weights_double);
			if(element->weights_float) {
				g_free(element->weights_float);
				element->weights_float = NULL;
			}
		} else
			channels = 0;
		element->weights_double = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &element->channels);

		/*
		 * if the number of channels has changed, force a caps
		 * renegotiation
		 */

		if(element->channels != channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SINK_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}

		g_mutex_unlock(element->weights_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_WEIGHTS:
		g_mutex_lock(element->weights_lock);
		if(element->weights_double)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->weights_double, element->channels));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		g_mutex_unlock(element->weights_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(object);

	g_mutex_free(element->weights_lock);
	element->weights_lock = NULL;
	g_free(element->weights_double);
	element->weights_double = NULL;
	g_free(element->weights_float);
	element->weights_float = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_sumsquares_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Sum-of-Squares", "Filter/Audio", "Computes the weighted sum-of-squares of the input channels.", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
}


/*
 * class_init()
 */


static void gstlal_sumsquares_class_init(GSTLALSumSquaresClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_WEIGHTS,
		g_param_spec_value_array(
			"weights",
			"Weights",
			"Vector of weights to use in sum.  If no vector is provided weights of 1.0 are assumed, otherwise the number of input channels must equal the vector length.  The incoming channels are first multiplied by the weights, then squared, then summed.",
			g_param_spec_double(
				"weight",
				"Weight",
				"Weight",
				-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_sumsquares_init(GSTLALSumSquares *filter, GSTLALSumSquaresClass *kclass)
{
	filter->channels = 0;
	filter->weights_lock = g_mutex_new();
	filter->weights_double = NULL;
	filter->weights_float = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}
