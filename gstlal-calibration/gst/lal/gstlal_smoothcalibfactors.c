/*
 * Copyright (C) 2009--2012,2014,2015, 2016  Kipp Cannon <kipp.cannon@ligo.org>, Madeline Wade <madeline.wade@ligo.org>
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
 * SECTION:gstlal_smoothcalibfactors
 * @short_description:  Smooths the calibration factors using a running median.
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
#include <gstlal_smoothcalibfactors.h>


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


#define GST_CAT_DEFAULT gstlal_smoothcalibfactors_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALSmoothCalibFactors,
	gstlal_smoothcalibfactors,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_smoothcalibfactors", 0, "lal_smoothcalibfactors element")
);


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

#define DEFINE_ELEM_SWAP_FUNC(DTYPE) \
static void elem_swap_## DTYPE(DTYPE a, DTYPE b) { register DTYPE t=(a);(a)=(b);(b)=t; }	

#define DEFINE_MEDIAN_FUNC(DTYPE) \
/*Algorithm from Numerical recipes in C of 1992*/ \
static DTYPE median_## DTYPE(DTYPE arr[], uint16_t n) {\
    uint16_t low, high ; \
    uint16_t median; \
    uint16_t middle, ll, hh; \
    low = 0 ; high = n-1 ; median = (low + high) / 2; \
    for (;;) { \
    	if (high <= low) /* One element only */ \
    		return arr[median] ; \
    	if (high == low + 1) { /* Two elements only */ \
    		if (arr[low] > arr[high]) \
    		elem_swap_## DTYPE(arr[low], arr[high]) ; \
    		return arr[median] ; \
    	} \
    	/* Find median of low, middle and high items; swap into position low */ \
    	middle = (low + high) / 2; \
   	 if (arr[middle] > arr[high]) \
    		elem_swap_## DTYPE(arr[middle], arr[high]) ; \
    	if (arr[low] > arr[high]) \
    		elem_swap_## DTYPE(arr[low], arr[high]) ; \
    	if (arr[middle] > arr[low]) \
    		elem_swap_## DTYPE(arr[middle], arr[low]) ; \
    	/* Swap low item (now in position middle) into position (low+1) */ \
    	elem_swap_## DTYPE(arr[middle], arr[low+1]) ; \
    	/* Nibble from each end towards middle, swapping items when stuck */ \
    	ll = low + 1; \
    	hh = high; \
    	for (;;) { \
    		do ll++; while (arr[low] > arr[ll]) ; \
    		do hh--; while (arr[hh] > arr[low]) ; \
    		if (hh < ll) \
    			break; \
	    	elem_swap_## DTYPE(arr[ll], arr[hh]) ; \
	    } \
    	/* Swap middle item (in position low) back into correct position */ \
    	elem_swap_## DTYPE(arr[low], arr[hh]) ; \
    	/* Re-set active partition */ \
    	if (hh <= median) \
    		low = ll; \
    	if (hh >= median) \
    		high = hh - 1; \
    } \
    return arr[median] ; \
} 
	

#define DEFINE_RUNNING_MEDIAN_FUNC(DTYPE) \
static DTYPE running_median_## DTYPE(GSTLALSmoothCalibFactors *element, DTYPE *src) { \
	int i; \
	if (element->med_array_size < element->med_array_size_max) {\
		*(element->fifo_array+(element->med_array_size)) = *src; \
		element->med_array_size += 1; \
	} \
	else { \
		for(i=0; i<((element->med_array_size)-1); i++) \
			element->fifo_array[i] = element->fifo_array[i+1]; \
		element->fifo_array[(element->med_array_size)-1] = *src; \
	} \
	uint16_t length = sizeof(element->fifo_array)/sizeof(element->fifo_array[0]); \
	return median_## DTYPE((DTYPE*) element->fifo_array, length); \
}
	

#define DEFINE_SMOOTH_FACTORS_FUNC(DTYPE) \
static GstFlowReturn smooth_factors_## DTYPE(GSTLALSmoothCalibFactors *element, GstMapInfo *inmap, GstMapInfo *outmap, gint n) { \
	DTYPE *src = (DTYPE *) inmap->data; \
	DTYPE *dst = (DTYPE *) outmap->data; \
	DTYPE *dst_end = dst + n; \
	for(; dst < dst_end; dst++) { \
		DTYPE *src_end = src + element->channels; \
		for(; src < src_end; src++) { \
			if (*src <= element->max_value && *src >= element->min_value && !(isnan(*src)) && !(isinf(*src))) { \
				if (element->statevector) \
					*dst = (DTYPE) 1.0; \
				else {\
					element->current_median_val = running_median_## DTYPE(element, src); \
					*dst = element->current_median_val; \
				} \
			} \
			else {\
				if (element->statevector) {\
					*dst = (DTYPE) 0.0; \
				} \
				else { \
					*dst = element->current_median_val; \
				} \
			} \
		} \
	} \
	return GST_FLOW_OK; \
}

DEFINE_ELEM_SWAP_FUNC(double);
DEFINE_ELEM_SWAP_FUNC(float);
DEFINE_MEDIAN_FUNC(double);
DEFINE_MEDIAN_FUNC(float);
DEFINE_RUNNING_MEDIAN_FUNC(double);
DEFINE_RUNNING_MEDIAN_FUNC(float);
DEFINE_SMOOTH_FACTORS_FUNC(double);
DEFINE_SMOOTH_FACTORS_FUNC(float);

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

/*
static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, element->channels, NULL);
		}
		break;

	case GST_PAD_SINK:

		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, element->channels, NULL);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}*/


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(trans);
	GstAudioInfo info;
	gboolean success = TRUE;

	/*
	 * parse the caps
	 */

	success = gst_audio_info_from_caps(&info, incaps);
	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse %" GST_PTR_FORMAT, incaps);

	/*
	 * set the median and smoothing functions
	 */

	if(success) {
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_F32:
			element->smooth_factors_func = smooth_factors_float;
			break;

		case GST_AUDIO_FORMAT_F64:
			element->smooth_factors_func = smooth_factors_double;
			break;

		default:
			GST_ERROR_OBJECT(element, "unsupported foramt %" GST_PTR_FORMAT, incaps);
			success = FALSE;
			break;
		}
	}

	if(success) {
		/* FIXME:  perhaps emit a "channel-count-changed" signal? */
		element->channels = GST_AUDIO_INFO_CHANNELS(&info);
	}

	return success;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(trans);
	GstMapInfo outmap;
	GstFlowReturn result;

	g_assert(element->smooth_factors_func != NULL);

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		GstMapInfo inmap;

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		result = element->smooth_factors_func(element, &inmap, &outmap, GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf));
		gst_buffer_unmap(inbuf, &inmap);
	} else {
		/*
		 * input is 0s.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(outmap.data, 0, outmap.size);
		result = GST_FLOW_OK;
	}

	gst_buffer_unmap(outbuf, &outmap);

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
	ARG_STATEVECTOR = 1,
	ARG_MED_ARRAY_SIZE_MAX,
	ARG_MAX_VALUE,
	ARG_MIN_VALUE,
	ARG_DEFAULT_OUT 
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MED_ARRAY_SIZE_MAX:
		element->med_array_size_max = g_value_get_int(value);
		break;
	case ARG_MAX_VALUE: 
		element->max_value = g_value_get_double(value);
		break;
	case ARG_MIN_VALUE:
		element->min_value = g_value_get_double(value);
		break;
	case ARG_STATEVECTOR:
		element->statevector = g_value_get_boolean(value);
		break;
	case ARG_DEFAULT_OUT:
		element->default_out = g_value_get_double(value);
		break; 

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MED_ARRAY_SIZE_MAX:
		g_value_set_int(value, element->med_array_size_max);
		break;
	case ARG_MAX_VALUE:
		g_value_set_double(value, element->max_value);
		break;
	case ARG_MIN_VALUE:
		g_value_set_double(value, element->min_value);
		break;
	case ARG_STATEVECTOR:
		g_value_set_boolean(value, element->statevector);
		break;
	case ARG_DEFAULT_OUT:
		g_value_set_double(value, element->default_out);
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
	GSTLALSmoothCalibFactors *element = GSTLAL_SMOOTHCALIBFACTORS(object);

        g_free(element->fifo_array);
        element->fifo_array = NULL;

	G_OBJECT_CLASS(gstlal_smoothcalibfactors_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE("{ " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) " }") ", " \
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


static void gstlal_smoothcalibfactors_class_init(GSTLALSmoothCalibFactorsClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Smooth Calibration Factors",
		"Filter/Audio",
		"Smooths the calibration factors with a running median and threshold cut.",
		"Madeline Wade <madeline.wade@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	//transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_MED_ARRAY_SIZE_MAX,
		g_param_spec_int(
			"max-size",
			"Maximum median array size",
			"Maximum size of the array of values from which the median is determiend",
			G_MININT, G_MAXINT, 1920,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAX_VALUE,
		g_param_spec_double(
			"max-value",
			"Maximum acceptable value",
			"Maximum acceptable value in order to be entered into array from which median is calculated.",
			-INFINITY, INFINITY, 1.2,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MIN_VALUE,
		g_param_spec_double(
			"min-value",
			"Minimum acceptable value",
			"Minimum acceptable value in order to be entered into array from which median is calculated.",
			-INFINITY, INFINITY, 0.8,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATEVECTOR,
		g_param_spec_boolean(
			"statevector",
			"Statevector mode",
			"Run the element in statevector mode where 1's or 0's are produced.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_OUT,
		g_param_spec_double(
			"default-val",
			"Default output value",
			"Default output value to be used if no input values pass min/max criteria.",
			-INFINITY, INFINITY, 1.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_smoothcalibfactors_init(GSTLALSmoothCalibFactors *filter)
{
	filter->channels = 0;
	filter->med_array_size = 0;
	filter->fifo_array = NULL;
	filter->current_median_val = filter->default_out;
	filter->smooth_factors_func = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}
