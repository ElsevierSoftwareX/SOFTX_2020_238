/*
 * Copyright (C) 2009--2012,2014,2015, 2016  Kipp Cannon <kipp.cannon@ligo.org>, Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>
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
 * SECTION:gstlal_smoothkappas
 * @short_description:  Smooths the calibration factors (kappas) using an
 * approximation to a running median.
 *
 * This element smooths the kappas using an approximation to a running
 * median of an array whose size is set by the property array-size. The
 * algorithm resets accepted unsmoothed kappas outside of a narrow range
 * to the min or max of that range. This range is centered on the last
 * smoothed kappa, and its width is determined by the property
 * kappa-ceiling. A new smoothed kappa is calculated as the mean of an 
 * array in which one element is the unsmoothed kappa and the rest are
 * copies of the previous smoothed kappa. 
 */


/*
 * ============================================================================
 *
 *				  Preamble
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
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_smoothkappas.h>


/*
 * ============================================================================
 *
 *				 Parameters
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) " }, " \
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
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) " }, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


#define GST_CAT_DEFAULT gstlal_smoothkappas_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALSmoothKappas,
	gstlal_smoothkappas,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_smoothkappas", 0, "lal_smoothkappas element")
);


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


#define DEFINE_GET_NEW_MEDIAN(DTYPE) \
static void get_new_median_ ## DTYPE( DTYPE *default_kappa, DTYPE max_kappa_offset, DTYPE kappa_ceiling, gint array_size, const DTYPE *unsmoothed_kappa, DTYPE *smoothed_kappa) { \
	if(*unsmoothed_kappa > *default_kappa - max_kappa_offset && *unsmoothed_kappa < *default_kappa + max_kappa_offset) { \
		if(*unsmoothed_kappa < *default_kappa - kappa_ceiling) \
			*default_kappa = (array_size * (*default_kappa) - kappa_ceiling ) / array_size; \
		else if(*unsmoothed_kappa > *default_kappa + kappa_ceiling) \
			*default_kappa = (array_size * (*default_kappa) + kappa_ceiling ) / array_size; \
		else \
			*default_kappa = ((array_size - 1) * (*default_kappa) + *unsmoothed_kappa) / array_size; \
	} \
	*smoothed_kappa = *default_kappa; \
}


#define DEFINE_MEDIAN_APPROX_FUNC(DTYPE) \
static void median_approx_func_ ## DTYPE(const DTYPE *src, DTYPE *dst, gint buffer_size, double *default_kappa, double max_kappa_offset, double kappa_ceiling, gint array_size, gboolean gap) { \
	if(!gap) { \
		for(gint i = 0; i < buffer_size; i++) { \
			get_new_median_ ## DTYPE((DTYPE *) default_kappa, (DTYPE) max_kappa_offset, (DTYPE) kappa_ceiling, array_size, src, dst); \
			src++; \
			dst++; \
		} \
	} else { \
		for(gint i = 0; i < buffer_size; i++) { \
			*dst = (DTYPE) *default_kappa; \
			dst++; \
		} \
	} \
}


DEFINE_GET_NEW_MEDIAN(float);
DEFINE_GET_NEW_MEDIAN(double);
DEFINE_MEDIAN_APPROX_FUNC(float);
DEFINE_MEDIAN_APPROX_FUNC(double);


/*
 * ============================================================================
 *
 *		     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);
	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);

	if(success) {
		*size = GST_AUDIO_INFO_BPF(&info);
		element->unit_size = *size;
	} else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);
	GstAudioInfo info;
	gboolean success = TRUE;

	/*
	 * parse the caps
	 */

	success = gstlal_audio_info_from_caps(&info, incaps);
	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse %" GST_PTR_FORMAT, incaps);

	/*
	 * set the median and smoothing functions
	 */

	if(success) {
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_F32:
			element->unit_size = 4;
			break;

		case GST_AUDIO_FORMAT_F64:
			element->unit_size = 8;
			break;

		default:
			GST_ERROR_OBJECT(element, "unsupported foramt %" GST_PTR_FORMAT, incaps);
			success = FALSE;
			break;
		}
	}

	return success;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);
	GstMapInfo inmap, outmap;

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	g_assert_cmpuint(inmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(outmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(inmap.size, ==, outmap.size);

	if(element->unit_size == 4) {
		gint buffer_size = outmap.size / element->unit_size;
		median_approx_func_float((const float *) inmap.data, (float *) outmap.data, buffer_size, &element->default_kappa, element->max_kappa_offset, element->kappa_ceiling, element->median_array_size, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP));
	} else if(element->unit_size == 8) {
		gint buffer_size = outmap.size / element->unit_size;
		median_approx_func_double((const double *) inmap.data, (double *) outmap.data, buffer_size, &element->default_kappa, element->max_kappa_offset, element->kappa_ceiling, element->median_array_size, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP));
	} else {
		g_assert_not_reached();
	}

	gst_buffer_unmap(outbuf, &outmap);
	gst_buffer_unmap(inbuf, &inmap);

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_MEDIAN_ARRAY_SIZE = 1,
	ARG_MAX_KAPPA_OFFSET,
	ARG_KAPPA_CEILING,
	ARG_DEFAULT_KAPPA
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MEDIAN_ARRAY_SIZE:
		element->median_array_size = g_value_get_int(value);
		break;
	case ARG_MAX_KAPPA_OFFSET: 
		element->max_kappa_offset = g_value_get_double(value);
		break;
	case ARG_KAPPA_CEILING:
		element->kappa_ceiling = g_value_get_double(value);
		break;
	case ARG_DEFAULT_KAPPA:
		element->default_kappa = g_value_get_double(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MEDIAN_ARRAY_SIZE:
		g_value_set_int(value, element->median_array_size);
		break;
	case ARG_MAX_KAPPA_OFFSET:
		g_value_set_double(value, element->max_kappa_offset);
		break;
	case ARG_KAPPA_CEILING:
		g_value_set_double(value, element->kappa_ceiling);
		break;
	case ARG_DEFAULT_KAPPA:
		g_value_set_double(value, element->default_kappa);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * class_init()
 */


static void gstlal_smoothkappas_class_init(GSTLALSmoothKappasClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Smooth Calibration Factors",
		"Filter/Audio",
		"Smooths the calibration factors with a running median and threshold cut.",
		"Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_MEDIAN_ARRAY_SIZE,
		g_param_spec_int(
			"array-size",
			"Median array size",
			"Size of the array of values from which the median is approximated",
			G_MININT, G_MAXINT, 2048,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAX_KAPPA_OFFSET,
		g_param_spec_double(
			"maximum-offset",
			"Maximum acceptable kappa offset",
			"Maximum acceptable offset of unsmoothed kappa from current median to be entered into array from which median is approximated.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.2,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_KAPPA_CEILING,
		g_param_spec_double(
			"kappa-ceiling",
			"Reset kappas outside of range",
			"Accepted unsmoothed kappas outside of the range [current-median-value - kappa-ceiling, current-median-value + kappa_ceiling], are reset to current-median-value +- kappa-ceiling",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.02,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_KAPPA,
		g_param_spec_double(
			"default-kappa",
			"Default kappa value",
			"Default kappa value to be used if no input values pass kappa-offset criteria and there is no recent good kappa value.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

}


/*
 * init()
 */


static void gstlal_smoothkappas_init(GSTLALSmoothKappas *element)
{
	element->unit_size = 0;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);

}
