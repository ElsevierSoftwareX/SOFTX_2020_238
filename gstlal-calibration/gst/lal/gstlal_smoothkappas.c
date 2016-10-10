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
#include <complex.h>


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
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
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
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
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


static void get_new_median(double new_element, double *fifo_array, double *current_median, gint array_size, int *index_re, int *index_im, gboolean array_is_imaginary) {
	if(array_is_imaginary) {
		fifo_array[*index_im] = new_element;
		if(*index_im < array_size - 1)
			(*index_im)++;
		else
			*index_im -= (array_size - 1);
	} else {
		fifo_array[*index_re] = new_element;
		if(*index_re < array_size - 1)
			(*index_re)++;
		else
			*index_re -= (array_size - 1);
	}

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

	return;
}


static double get_average(double new_element, double *fifo_array, gint array_size, int *index_re, int *index_im, gboolean array_is_imaginary) {
	if(array_is_imaginary) {
		fifo_array[*index_im] = new_element;
		if(*index_im < array_size - 1)
			(*index_im)++;
		else
			*index_im -= (array_size - 1);
	} else {
		fifo_array[*index_re] = new_element;
		if(*index_re < array_size - 1)
			(*index_re)++;
		else
			*index_re -= (array_size - 1);
	}
	double sum = 0;
	int i;
	for(i = 0; i < array_size; i++) {
		sum += fifo_array[i];
	}
	return sum / array_size;
}


#define DEFINE_SMOOTH_BUFFER(DTYPE) \
static GstFlowReturn smooth_buffer_ ## DTYPE(const DTYPE *src, DTYPE *dst, gint buffer_size, double *fifo_array, double *avg_array, double default_kappa, double *current_median, double maximum_offset, gint array_size, gint avg_array_size, int *index_re, int *index_im, int *avg_index_re, int *avg_index_im, gboolean gap, gboolean default_to_median, gboolean track_bad_kappa) { \
	gint i; \
	double new_element; \
	for(i = 0; i < buffer_size; i++) { \
		if(gap || (double) *src > default_kappa + maximum_offset || (double) *src < default_kappa - maximum_offset || isnan(*src) || isinf(*src) || (double) *src == 0.0) { \
			if(default_to_median) \
				new_element = *current_median; \
			else \
				new_element = default_kappa; \
		} else \
			new_element = (double) *src; \
 \
		get_new_median(new_element, fifo_array, current_median, array_size, index_re, index_im, FALSE); \
		*dst = (DTYPE) get_average(*current_median, avg_array, avg_array_size, avg_index_re, avg_index_im, FALSE); \
		if (track_bad_kappa) { \
                        if (*dst == default_kappa) \
                                *dst = 0.0; \
                        else \
                                *dst = 1.0; \
		} \
		src++; \
		dst++; \
	} \
	return GST_FLOW_OK; \
}


#define DEFINE_SMOOTH_COMPLEX_BUFFER(DTYPE) \
static GstFlowReturn smooth_complex_buffer_ ## DTYPE(const DTYPE complex *src, DTYPE complex *dst, gint buffer_size, double *fifo_array_re, double *fifo_array_im, double *avg_array_re, double *avg_array_im, double default_kappa_re, double default_kappa_im, double *current_median_re, double *current_median_im, double maximum_offset_re, double maximum_offset_im, gint array_size, gint avg_array_size, int *index_re, int *index_im, int *avg_index_re, int *avg_index_im, gboolean gap, gboolean default_to_median, gboolean track_bad_kappa) { \
	gint i; \
	double new_element_re, new_element_im; \
	for(i = 0; i < buffer_size; i++) { \
		double complex doublesrc = (double complex) *src; \
		if(gap || creal(doublesrc) > default_kappa_re + maximum_offset_re || creal(doublesrc) < default_kappa_re - maximum_offset_re || isnan(creal(doublesrc)) || isinf(creal(doublesrc)) || creal(doublesrc) == 0) { \
			if(default_to_median) \
				new_element_re = *current_median_re; \
			else \
				new_element_re = default_kappa_re; \
		} else { \
			new_element_re = creal(doublesrc); \
		} \
		if(gap || cimag(doublesrc) > default_kappa_im + maximum_offset_im || cimag(doublesrc) < default_kappa_im - maximum_offset_im || isnan(cimag(doublesrc)) || isinf(cimag(doublesrc)) || cimag(doublesrc) == 0) { \
			if(default_to_median) \
				new_element_im = *current_median_im; \
			else \
				new_element_im = default_kappa_im; \
		} else { \
			new_element_im = cimag(doublesrc); \
		} \
		get_new_median(new_element_re, fifo_array_re, current_median_re, array_size, index_re, index_im, FALSE); \
		get_new_median(new_element_im, fifo_array_im, current_median_im, array_size, index_re, index_im, TRUE); \
		*dst = (DTYPE) get_average(*current_median_re, avg_array_re, avg_array_size, avg_index_re, avg_index_im, FALSE) + I * (DTYPE) get_average(*current_median_im, avg_array_im, avg_array_size, avg_index_re, avg_index_im, TRUE); \
		if(track_bad_kappa) { \
			if((creal((double complex) *dst) == default_kappa_re) && (cimag((double complex) *dst) == default_kappa_im)) \
				*dst = 0.0; \
			else if(cimag((double complex) *dst) == default_kappa_im) \
				*dst = 1.0; \
			else if(creal((double complex) *dst) == default_kappa_re) \
				*dst = I; \
			else \
				*dst = 1.0 + I; \
		} \
		src++; \
		dst++; \
	} \
	return GST_FLOW_OK; \
}


DEFINE_SMOOTH_BUFFER(float);
DEFINE_SMOOTH_BUFFER(double);
DEFINE_SMOOTH_COMPLEX_BUFFER(float);
DEFINE_SMOOTH_COMPLEX_BUFFER(double);


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
	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);

	if(success) {
		*size = GST_AUDIO_INFO_BPF(&info);
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
	gboolean success = TRUE;
	gsize unit_size;

	/*
	 * parse the caps
	 */

	success &= get_unit_size(trans, incaps, &unit_size);
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);

	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse caps %" GST_PTR_FORMAT, incaps);

	/*
	 * record stream parameters
	 */

	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_SMOOTHKAPPAS_F32;
			g_assert_cmpuint(unit_size, ==, 4);
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_SMOOTHKAPPAS_F64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_SMOOTHKAPPAS_Z64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_SMOOTHKAPPAS_Z128;
			g_assert_cmpuint(unit_size, ==, 16);
		} else
			g_assert_not_reached();

		element->unit_size = unit_size;
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALSmoothKappas *element = GSTLAL_SMOOTHKAPPAS(trans);

	element->current_median_re = element->default_kappa_re;
	element->current_median_im = element->default_kappa_im;

	element->fifo_array_re = g_malloc(sizeof(double) * element->array_size);
	element->fifo_array_im = g_malloc(sizeof(double) * element->array_size);
	element->avg_array_re = g_malloc(sizeof(double) * element->avg_array_size);
	element->avg_array_im = g_malloc(sizeof(double) * element->avg_array_size);

	int i;
	for(i = 0; i < element->array_size; i++, (element->fifo_array_re)++, (element->fifo_array_im)++) { 
		*(element->fifo_array_re) = element->default_kappa_re;
		*(element->fifo_array_im) = element->default_kappa_im;
	}
	for(i = 0; i < element->avg_array_size; i++, (element->avg_array_re)++, (element->avg_array_im)++) {
		*(element->avg_array_re) = element->default_kappa_re;
		*(element->avg_array_im) = element->default_kappa_im;
	}

	(element->fifo_array_re) -= element->array_size;
	(element->fifo_array_im) -= element->array_size;

	(element->avg_array_re) -= element->avg_array_size;
	(element->avg_array_im) -= element->avg_array_size;

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

	if(element->data_type == GSTLAL_SMOOTHKAPPAS_F32) {
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_buffer_float((const float *) inmap.data, (float *) outmap.data, buffer_size, element->fifo_array_re, element->avg_array_re, element->default_kappa_re, &element->current_median_re, element->maximum_offset_re, element->array_size, element->avg_array_size, &element->index_re, &element->index_im, &element->avg_index_re, &element->avg_index_im, gap, element->default_to_median, element->track_bad_kappa);
	} else if(element->data_type == GSTLAL_SMOOTHKAPPAS_F64) {
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_buffer_double((const double *) inmap.data, (double *) outmap.data, buffer_size, element->fifo_array_re, element->avg_array_re, element->default_kappa_re, &element->current_median_re, element->maximum_offset_re, element->array_size, element->avg_array_size, &element->index_re, &element->index_im, &element->avg_index_re, &element->avg_index_im, gap, element->default_to_median, element->track_bad_kappa);
	} else if(element->data_type == GSTLAL_SMOOTHKAPPAS_Z64) {
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_complex_buffer_float((const float complex *) inmap.data, (float complex *) outmap.data, buffer_size, element->fifo_array_re, element->fifo_array_im, element->avg_array_re, element->avg_array_im, element->default_kappa_re, element->default_kappa_im, &element->current_median_re, &element->current_median_im, element->maximum_offset_re, element->maximum_offset_im, element->array_size, element->avg_array_size, &element->index_re, &element->index_im, &element->avg_index_re, &element->avg_index_im, gap, element->default_to_median, element->track_bad_kappa);
	} else if(element->data_type == GSTLAL_SMOOTHKAPPAS_Z128) { 
		gint buffer_size = outmap.size / element->unit_size;
		result = smooth_complex_buffer_double((const double complex *) inmap.data, (double complex *) outmap.data, buffer_size, element->fifo_array_re, element->fifo_array_im, element->avg_array_re, element->avg_array_im, element->default_kappa_re, element->default_kappa_im, &element->current_median_re, &element->current_median_im, element->maximum_offset_re, element->maximum_offset_im, element->array_size, element->avg_array_size, &element->index_re, &element->index_im, &element->avg_index_re, &element->avg_index_im, gap, element->default_to_median, element->track_bad_kappa);
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
	ARG_AVG_ARRAY_SIZE,
	ARG_DEFAULT_KAPPA_RE,
	ARG_DEFAULT_KAPPA_IM,
	ARG_MAXIMUM_OFFSET_RE,
	ARG_MAXIMUM_OFFSET_IM,
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
	case ARG_AVG_ARRAY_SIZE:
		element->avg_array_size = g_value_get_int(value);
		break;
	case ARG_DEFAULT_KAPPA_RE:
		element->default_kappa_re = g_value_get_double(value);
		break;
	case ARG_DEFAULT_KAPPA_IM:
		element->default_kappa_im = g_value_get_double(value);
		break;
	case ARG_MAXIMUM_OFFSET_RE:
		element->maximum_offset_re = g_value_get_double(value);
		break;
	case ARG_MAXIMUM_OFFSET_IM:
		element->maximum_offset_im = g_value_get_double(value);
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
	case ARG_AVG_ARRAY_SIZE:
		g_value_set_int(value, element->avg_array_size);
		break;
	case ARG_DEFAULT_KAPPA_RE:
		g_value_set_double(value, element->default_kappa_re);
		break;
	case ARG_DEFAULT_KAPPA_IM:
		g_value_set_double(value, element->default_kappa_im);
		break;
	case ARG_MAXIMUM_OFFSET_RE:
		g_value_set_double(value, element->maximum_offset_re);
		break;
	case ARG_MAXIMUM_OFFSET_IM:
		g_value_set_double(value, element->maximum_offset_im);
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
	g_free(element->fifo_array_re);
	element->fifo_array_re = NULL;
	g_free(element->fifo_array_im);
	element->fifo_array_im = NULL;
	g_free(element->avg_array_re);
	element->avg_array_re = NULL;
	g_free(element->avg_array_im);
	element->avg_array_im = NULL;
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
		ARG_AVG_ARRAY_SIZE,
		g_param_spec_int(
			"avg-array-size",
			"Average array size",
			"Size of the array of values from which the average is calculated\n\t\t\t"
			"from the median values. By default, no average is taken.",
			G_MININT, G_MAXINT, 1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_KAPPA_RE,
		g_param_spec_double(
			"default-kappa-re",
			"Default real part of kappa value",
			"Default real part of kappa value to be used if there is a gap in the\n\t\t\t"
			"incoming buffer, or if no input values pass kappa-offset criteria.\n\t\t\t"
			"All elements of the real fifo array are initialized to this value.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_KAPPA_IM,
		g_param_spec_double(
			"default-kappa-im",
			"Default imaginary part of kappa value",
			"Default imaginary part of kappa value to be used if there is a gap in the\n\t\t\t"
			"incoming buffer, or if no input values pass kappa-offset criteria. All\n\t\t\t"
			"elements of the imaginary fifo array are initialized to this value.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAXIMUM_OFFSET_RE,
		g_param_spec_double(
			"maximum-offset-re",
			"Maximum acceptable real kappa offset",
			"Maximum acceptable offset of unsmoothed real kappa from default-kappa-re\n\t\t\t"
			"to be entered into real array from which median is calculated.",
			0, G_MAXDOUBLE, G_MAXDOUBLE / 2,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAXIMUM_OFFSET_IM,
		g_param_spec_double(
			"maximum-offset-im",
			"Maximum acceptable imaginary kappa offset",
			"Maximum acceptable offset of unsmoothed imaginary kappa from default-kappa-im\n\t\t\t"
			"to be entered into imaginary-part array from which median is calculated.",
			0, G_MAXDOUBLE, G_MAXDOUBLE / 2,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_TO_MEDIAN,
		g_param_spec_boolean(
			"default-to-median",
			"Default to median",
			"If set to false (default), gaps (or times where input values do not pass\n\t\t\t"
			"kappa-offset criteria) are filled in by entering default-kappa into the\n\t\t\t"
			"fifo array. If set to true, gaps are filled in by entering the current\n\t\t\t"
			"median value into the fifo array.",
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
			"If set to false (default), gaps (or times where input values do not pass\n\t\t\t"
			"kappa-offset criteria) are filled in by entering default-kappa into the fifo\n\t\t\t"
			"array and non-gaps use the input buffer value. If set to true, gaps are\n\t\t\t"
			"filled in by entering 0 into the fifo array and non-gaps are filled by\n\t\t\t"
			"entering 1's into the fifo array.",
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
	element->avg_array_size = 0;
	element->fifo_array_re = NULL;
	element->fifo_array_im = NULL;
	element->avg_array_re = NULL;
	element->avg_array_im = NULL;
	element->index_re = 0;
	element->index_im = 0;
	element->avg_index_re = 0;
	element->avg_index_im = 0;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}
