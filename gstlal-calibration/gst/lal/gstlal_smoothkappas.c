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
 * @short_description:  Smooths the calibration factors (kappas) using a 
 * running median.
 *
 * This element smooths the kappas using a running median of an array 
 * whose size is set by the property array-size. When a new raw value
 * is entered into the array, it replaces the oldest value in the array
 * (first in, first out). When this element receives a gap as input, it 
 * will output a default kappa value (set by the property default-kappa)
 * until it receives a buffer that is not flagged as a gap.
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


static double get_new_median(double new_element, double *fifo_array, double *current_median, gint array_size) {
	static int i;
	if(!i)
		i = 0;
	fifo_array[i] = new_element;
	if(i < array_size - 1)
		i++;
	else
		i -= (array_size - 1);

	int j, number_less, number_greater, number_equal;
	double first_greater, second_greater, first_less, second_less;
	number_less = 0;
	number_equal = 0;
	number_greater = 0;
	first_greater = G_MAXDOUBLE;
	second_greater = G_MAXDOUBLE;
	first_less = -G_MAXDOUBLE;
	second_less = -G_MAXDOUBLE;
	for(j = 0; j < array_size; j++) {
		if(fifo_array[j] < *current_median) {
			number_less++;
			if(fifo_array[j] > first_less) {
				second_less = first_less;
				first_less = fifo_array[j];
			} else if(fifo_array[j] > second_less)
				second_less = fifo_array[j];
		}
		else if(fifo_array[j] == *current_median)
			number_equal++;
		else if(fifo_array[j] > *current_median) {
			number_greater++;
			if(fifo_array[j] < first_greater) {
				second_greater = first_greater;
				first_greater = fifo_array[j];
			} else if(fifo_array[j] < second_greater)
				second_greater = fifo_array[j];
		}
		else
			g_assert_not_reached();
	}

	g_assert_cmpint(number_less + number_equal + number_greater, ==, array_size);

	if((!(array_size % 2)) && (number_less == array_size / 2) && (number_greater == array_size / 2))
		*current_median = (first_greater + first_less) / 2;
	else if((!(array_size % 2)) && (number_greater > array_size / 2))
		*current_median = (first_greater + second_greater) / 2;
	else if((!(array_size % 2)) && (number_less > array_size / 2))
		*current_median = (first_less + second_less) / 2;
	else if((!(array_size % 2)) && (number_greater == array_size / 2) && (number_less < array_size / 2))
		*current_median = (*current_median + first_greater) / 2;
	else if((!(array_size % 2)) && (number_less == array_size / 2) && (number_greater < array_size / 2))
		*current_median = (*current_median + first_less) / 2;
	else if((array_size % 2) && (number_greater > array_size / 2))
		*current_median = first_greater;
	else if((array_size % 2) && (number_less > array_size / 2))
		*current_median = first_less;

	return *current_median;
}


#define DEFINE_SMOOTH_BUFFER(DTYPE) \
static GstFlowReturn smooth_buffer_ ## DTYPE(const DTYPE *src, DTYPE *dst, gint buffer_size, double *fifo_array, double default_kappa, double *current_median, double maximum_offset, gint array_size, gboolean gap, gboolean default_to_median, gboolean track_bad_kappa) { \
	gint i; \
	DTYPE new_element; \
	for(i = 0; i < buffer_size; i++) { \
		if(gap || (double) *src > default_kappa + maximum_offset || (double) *src < default_kappa - maximum_offset) { \
			if(default_to_median) \
				new_element = *current_median; \
			else if (track_bad_kappa) \
				new_element = 0.0; \
			else \
				new_element = default_kappa; \
		} else \
			if (track_bad_kappa) \
				new_element = 1.0; \
			else \
				new_element = *src; \
		*dst = (DTYPE) get_new_median((double) new_element, fifo_array, current_median, array_size); \
		src++; \
		dst++; \
	} \
	return GST_FLOW_OK; \
}


DEFINE_SMOOTH_BUFFER(float);
DEFINE_SMOOTH_BUFFER(double);


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
		GST_ERROR_OBJECT(element, "unable to parse caps %" GST_PTR_FORMAT, incaps);

	/*
	 * set the unit size
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
			GST_ERROR_OBJECT(element, "unsupported format %" GST_PTR_FORMAT, incaps);
			success = FALSE;
			break;
		}
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);

	element->current_median = element->default_kappa;

	int i;
	element->fifo_array = g_malloc(sizeof(double) * element->array_size);

	for(i = 0; i < element->array_size; i++, (element->fifo_array)++) 
		*(element->fifo_array) = element->default_kappa;

	(element->fifo_array) -= element->array_size;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result;

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	gboolean gap = GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP);

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);

	g_assert_cmpuint(inmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(outmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(inmap.size, ==, outmap.size);

	if(element->unit_size == 4) {
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_buffer_float((const float *) inmap.data, (float *) outmap.data, buffer_size, element->fifo_array, element->default_kappa, &element->current_median, element->maximum_offset, element->array_size, gap, element->default_to_median, element->track_bad_kappa);
	} else if(element->unit_size == 8) {
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_buffer_double((const double *) inmap.data, (double *) outmap.data, buffer_size, element->fifo_array, element->default_kappa, &element->current_median, element->maximum_offset, element->array_size, gap, element->default_to_median, element->track_bad_kappa);
	} else {
		g_assert_not_reached();
	}

	GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);

	gst_buffer_unmap(inbuf, &inmap);

	gst_buffer_unmap(outbuf, &outmap);

	/*
	 * done
	 */

	return result;
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
	ARG_ARRAY_SIZE = 1,
	ARG_DEFAULT_KAPPA,
	ARG_MAXIMUM_OFFSET,
	ARG_DEFAULT_TO_MEDIAN,
	ARG_TRACK_BAD_KAPPA
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_ARRAY_SIZE:
		element->array_size = g_value_get_int(value);
		break;
	case ARG_DEFAULT_KAPPA:
		element->default_kappa = g_value_get_double(value);
		break;
	case ARG_MAXIMUM_OFFSET:
		element->maximum_offset = g_value_get_double(value);
		break;
	case ARG_DEFAULT_TO_MEDIAN:
		element->default_to_median = g_value_get_boolean(value);
		break;
	case ARG_TRACK_BAD_KAPPA:
		element->track_bad_kappa = g_value_get_boolean(value);
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
	case ARG_ARRAY_SIZE:
		g_value_set_int(value, element->array_size);
		break;
	case ARG_DEFAULT_KAPPA:
		g_value_set_double(value, element->default_kappa);
		break;
	case ARG_MAXIMUM_OFFSET:
		g_value_set_double(value, element->maximum_offset);
		break;
	case ARG_DEFAULT_TO_MEDIAN:
		g_value_set_boolean(value, element->default_to_median);
		break;
	case ARG_TRACK_BAD_KAPPA:
		g_value_set_boolean(value, element->track_bad_kappa);
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
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(object);
	g_free(element->fifo_array);
	element->fifo_array = NULL;
	G_OBJECT_CLASS(gstlal_smoothkappas_parent_class)->finalize(object);
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
		"Smooths the calibration factors with a running median.",
		"Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_ARRAY_SIZE,
		g_param_spec_int(
			"array-size",
			"Median array size",
			"Size of the array of values from which the median is calculated",
			G_MININT, G_MAXINT, 2048,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_KAPPA,
		g_param_spec_double(
			"default-kappa",
			"Default kappa value",
			"Default kappa value to be used if there is a gap in the incoming buffer, or if no input values pass kappa-offset criteria. All elements of the fifo array are initialized to this value.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAXIMUM_OFFSET,
		g_param_spec_double(
			"maximum-offset",
			"Maximum acceptable kappa offset",
			"Maximum acceptable offset of unsmoothed kappa from default-kappa to be entered into array from which median is calculated.",
			0, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_TO_MEDIAN,
		g_param_spec_boolean(
			"default-to-median",
			"Default to median",
			"If set to false (default), gaps (or times where input values do not pass kappa-offset criteria) are filled in by entering default-kappa into the fifo array. If set to true, gaps are filled in by entering the current median value into the fifo array.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TRACK_BAD_KAPPA,
		g_param_spec_boolean(
			"track-bad-kappa",
			"Track input bad kappas",
			"If set to false (default), gaps (or times where input values do not pass kappa-offset criteria) are filled in by entering default-kappa into the fifo array and non-gaps use the input buffer value. If set to true, gaps are filled in by entering 0 into the fifo array and non-gaps are filled by entering 1's into the fifo array.",
			FALSE,
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
	element->array_size = 0;
	element->fifo_array = NULL;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}
