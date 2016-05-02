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
 * SECTION:gstlal_sumsquares
 * @short_description:  Computes the weighted sum-of-squares of the input channels.
 *
 * Computes the weighted sum-of-squares of the input channels.
 *
 * Reviewed:  fcba8806c67cc57e279f3b1f5d2706cd355f35c8 2014-08-10 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Completed Actions:
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
 * stuff from C
 */


#include <math.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_sumsquares.h>


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_sumsquares_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALSumSquares,
	gstlal_sumsquares,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_sumsquares", 0, "lal_sumsquares element")
);


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


#define DEFINE_MAKE_WEIGHTS_NATIVE_FUNC(DTYPE) \
static void *make_weights_native_ ## DTYPE(GSTLALSumSquares *element) \
{ \
	DTYPE *weights_native = g_malloc(element->channels * sizeof(*weights_native)); \
	gint i; \
 \
	if(weights_native) \
		for(i = 0; i < element->channels; i++) \
			weights_native[i] = element->weights[i]; \
 \
	return weights_native; \
}


#define DEFINE_SUMSQUARES_FUNC(DTYPE) \
static GstFlowReturn sumsquares_ ## DTYPE(GSTLALSumSquares *element, GstBuffer *inbuf, GstBuffer *outbuf) \
{ \
	GstMapInfo in_info, out_info; \
	const DTYPE *src; \
	DTYPE *dst; \
	DTYPE *dst_end; \
\
	gst_buffer_map(inbuf, &in_info, GST_MAP_READ); \
	src = (const DTYPE *) in_info.data; \
	gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE); \
	dst = (DTYPE *) out_info.data; \
	dst_end = dst + (GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf)); \
 \
	if(element->weights_native) { \
		for(; dst < dst_end; dst++) { \
			const DTYPE *src_end = src + element->channels; \
			const DTYPE *w = element->weights_native; \
			for(*dst = 0; src < src_end; w++, src++) { \
				DTYPE x = *w * *src; \
				*dst += x * x; \
			} \
		} \
	} else { \
		for(; dst < dst_end; dst++) { \
			const DTYPE *src_end = src + element->channels; \
			for(*dst = 0; src < src_end; src++) \
				*dst += *src * *src; \
		} \
	} \
 \
	gst_buffer_unmap(inbuf, &in_info); \
	gst_buffer_unmap(outbuf, &out_info); \
	return GST_FLOW_OK; \
}


DEFINE_MAKE_WEIGHTS_NATIVE_FUNC(double)
DEFINE_MAKE_WEIGHTS_NATIVE_FUNC(float)
DEFINE_SUMSQUARES_FUNC(double)
DEFINE_SUMSQUARES_FUNC(float)


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
	gboolean success = TRUE;

	success &= gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
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

		g_mutex_lock(&element->weights_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->weights)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, element->channels, NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(&element->weights_lock);
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
	GstAudioInfo info;

	/*
	 * parse the caps
	 */

	if(!gst_audio_info_from_caps(&info, incaps)) {
		GST_ERROR_OBJECT(element, "unable to parse caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/*
	 * if applicable, check that the number of channels is valid
	 */

	g_mutex_lock(&element->weights_lock);
	if(!element->weights)
		element->channels = GST_AUDIO_INFO_CHANNELS(&info);
	else if(GST_AUDIO_INFO_CHANNELS(&info) != element->channels) {
		/* FIXME:  perhaps emit a "channel-count-changed" signal? */
		GST_ERROR_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, element->channels, incaps);
		g_mutex_unlock(&element->weights_lock);
		return FALSE;
	}

	/*
	 * set the sum-of-squares function
	 */

	switch(GST_AUDIO_INFO_WIDTH(&info)) {
	case 32:
		element->make_weights_native_func = make_weights_native_float;
		element->sumsquares_func = sumsquares_float;
		break;

	case 64:
		element->make_weights_native_func = make_weights_native_double;
		element->sumsquares_func = sumsquares_double;
		break;

	default:
		g_assert_not_reached();
		break;
	}

	/*
	 * force rebuild of weights in native data type
	 */

	g_free(element->weights_native);
	element->weights_native = NULL;
	g_mutex_unlock(&element->weights_lock);

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(trans);
	GstFlowReturn result;
	GstMapInfo out_info;

	g_assert(element->sumsquares_func != NULL);

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		g_mutex_lock(&element->weights_lock);
		if(element->weights && !element->weights_native) {
			element->weights_native = element->make_weights_native_func(element);
			g_assert(element->weights_native != NULL);
		}
		result = element->sumsquares_func(element, inbuf, outbuf);
		g_mutex_unlock(&element->weights_lock);
	} else {
		/*
		 * input is 0s.
		 */

		gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE);
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(out_info.data, 0, out_info.size);
		gst_buffer_unmap(outbuf, &out_info);
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
 * properties
 */


enum property {
	ARG_WEIGHTS = 1
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_WEIGHTS: {
		gint channels;
		g_mutex_lock(&element->weights_lock);

		/*
		 * record the old number of channels, and extract the
		 * weights
		 */

		if(element->weights) {
			channels = element->channels;
			g_free(element->weights);
		} else
			channels = 0;
		element->weights = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &element->channels);

		/*
		 * force rebuild of weights in native data type
		 */

		g_free(element->weights_native);
		element->weights_native = NULL;

		/*
		 * if the number of channels has changed, force a caps
		 * renegotiation
		 */

		if(element->channels != channels) {
			gst_base_transform_reconfigure_sink(GST_BASE_TRANSFORM(object));
		}

		g_mutex_unlock(&element->weights_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALSumSquares *element = GSTLAL_SUMSQUARES(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_WEIGHTS:
		g_mutex_lock(&element->weights_lock);
		if(element->weights)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->weights, element->channels));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		g_mutex_unlock(&element->weights_lock);
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

	g_mutex_clear(&element->weights_lock);
	g_free(element->weights);
	element->weights = NULL;
	g_free(element->weights_native);
	element->weights_native = NULL;

	G_OBJECT_CLASS(gstlal_sumsquares_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static void gstlal_sumsquares_class_init(GSTLALSumSquaresClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Sum-of-Squares",
		"Filter/Audio",
		"Computes the weighted sum-of-squares of the input channels.",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

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


static void gstlal_sumsquares_init(GSTLALSumSquares *filter)
{
	filter->channels = 0;
	g_mutex_init(&filter->weights_lock);
	filter->weights = NULL;
	filter->weights_native = NULL;
	filter->make_weights_native_func = NULL;
	filter->sumsquares_func = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}
