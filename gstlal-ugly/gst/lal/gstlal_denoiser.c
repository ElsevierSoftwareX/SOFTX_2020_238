/*
 * Copyright (C) 2020 Patrick Godwin, Chad Hanna
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
 * SECTION:gstlal_denoiser
 * @short_description:  Separate out stationary/non-stationary components from signals.
 *
 * Separate out stationary/non-stationary components from signals.
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
 *  stuff from gsl
 */


#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_vector.h>


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
#include <gstlal_denoiser.h>


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


/* Need to define "property" before the GObject Methods section since it gets used earlier */
enum property {
	ARG_STATIONARY = 1,
	ARG_THRESHOLD = 2,
	ARG_FAKE
};

static GParamSpec *properties[ARG_FAKE];

#define DEFAULT_STATIONARY FALSE
#define DEFAULT_THRESHOLD 1.0


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_denoiser_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALDenoiser,
	gstlal_denoiser,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_denoiser", 0, "lal_denoiser element")
);


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


float erfinv32(float x) {
	// based on https://stackoverflow.com/a/40260471
	float tt1, tt2, lnx, sgn;
	sgn = (x < 0) ? -1.0f : 1.0f;

	x = (1 - x)*(1 + x);  // x = 1 - x*x;
	lnx = logf(x);

	tt1 = 2/(M_PI*0.147) + 0.5f * lnx;
	tt2 = 1/(0.147) * lnx;

	return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}


double erfinv64(double x) {
	// based on https://stackoverflow.com/a/40260471
	double tt1, tt2, lnx, sgn;
	sgn = (x < 0) ? -1.0f : 1.0f;

	x = (1 - x)*(1 + x);  // x = 1 - x*x;
	lnx = logf(x);

	tt1 = 2/(M_PI*0.147) + 0.5f * lnx;
	tt2 = 1/(0.147) * lnx;

	return(sgn*sqrtf(-tt1 + sqrtf(tt1*tt1 - tt2)));
}


void orthogonalize32(gsl_vector_float *v, gsl_vector_float *x, gsl_vector_float *u1, gsl_vector_float *u2) {
	float u1u1, u2u2, u1v, u2v;
	gsl_vector_float *vsub = gsl_vector_float_alloc(x->size);
	gsl_vector_float_memcpy(u1, x);
	gsl_vector_float_memcpy(u2, v);
	gsl_vector_float_memcpy(vsub, u1);

	// set up basis vector u2
	gsl_blas_s_dot(u1, u1, u1u1);
	gsl_blas_s_dot(u1, v, u1v);
	gsl_vector_float_scale(vsub, u1v / u1u1);
	gsl_vector_float_sub(u2, vsub);
	gsl_blas_s_dot(u2, u2, u2u2);

	// normalize
	gsl_vector_float_scale(u1, 1. / sqrt(u1u1));
	gsl_vector_float_scale(u2, 1. / sqrt(u2u2));

	// get parallel (u1) and perpendicular (u2) components
	gsl_blas_s_dot(u1, v, u1v);
	gsl_vector_float_scale(u1, u1v);
	gsl_blas_s_dot(u2, v, u2v);
	gsl_vector_float_scale(u2, u2v);

	gsl_vector_float_free(vsub);

}


void orthogonalize64(gsl_vector *v, gsl_vector *x, gsl_vector *u1, gsl_vector *u2) {
	double u1u1, u2u2, u1v, u2v;
	gsl_vector *vsub = gsl_vector_alloc(x->size);
	gsl_vector_memcpy(u1, x);
	gsl_vector_memcpy(u2, v);
	gsl_vector_memcpy(vsub, u1);

	// set up basis vector u2
	gsl_blas_d_dot(u1, u1, u1u1);
	gsl_blas_d_dot(u1, v, u1v);
	gsl_vector_scale(vsub, u1v / u1u1);
	gsl_vector_sub(u2, vsub);
	gsl_blas_d_dot(u2, u2, u2u2);

	// normalize
	gsl_vector_scale(u1, 1. / sqrt(u1u1));
	gsl_vector_scale(u2, 1. / sqrt(u2u2));

	// get parallel (u1) and perpendicular (u2) components
	gsl_blas_d_dot(u1, v, u1v);
	gsl_vector_scale(u1, u1v);
	gsl_blas_d_dot(u2, v, u2v);
	gsl_vector_scale(u2, u2v);

	gsl_vector_free(vsub);

}


static GstFlowReturn denoise32(GstBuffer *inbuf, GstBuffer *outbuf, gboolean stationary, gdouble threshold) {

	GstMapInfo in_info, out_info;
	const float *src;
	float *dst;
	int i;
	size_t n = sizeof(src) / sizeof(float);
	size_t *ix[n];
	gsl_vector_float *npbar = gsl_vector_float_alloc(n);
	gsl_vector_float *gpbar = gsl_vector__floatalloc(n);
	gsl_vector_float_view src_view = gsl_vector_float_view_array(src, n);

	gst_buffer_map(inbuf, &in_info, GST_MAP_READ);
	src = (const float *) in_info.data;
	gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE);
	dst = (float *) out_info.data;

	// switch basis to be ordered by magnitude of data
	gsl_sort_float_index(ix, src, 1, n);
	gsl_permute_float(ix, src, 1, n);

	// calculate the average expected noise in the new basis
	for (i = 1; i < (n+1); i++) {
		gsl_vector_float_set(npbar, i, i / n);
	}
	gsl_vector_float_scale(npbar, 2);
	gsl_vector_float_add_constant(npbar, -1);
	for (i = 0; i < n; i++) {
		gsl_vector_float_set(npbar, i, erfinv64(gsl_vector_float_get(npbar, i)));
	}
	gsl_vector_float_scale(npbar, M_SQRT2);

	// remove average expected noise
	gsl_vector_float_memcpy(gpbar, &src_view.vector);
	gsl_vector_float_sub(gpbar, npbar);

	// keep non-stationary noise in up to predetermined threshold
	size_t half = n / 2;
	gsl_vector_float_view left = gsl_vector_float_subvector(gpbar, 0, half);
	gsl_vector_float_view right = gsl_vector_float_subvector(gpbar, half, n - half);
	for (i = 0; i < half; i++) {
		if (gsl_vector_float_get(&left.vector, i) > -threshold)
			gsl_vector_float_set(&left.vector, i, 0);
	}
	for (i = 0; i < n - half; i++) {
		if (gsl_vector_float_get(&right.vector, i) < threshold)
			gsl_vector_float_set(&right.vector, i, 0);
	}

	// decompose into stationary/non-stationary components
	if (!gsl_vector_float_isnull(gpbar)) {
		gsl_permute_inverse_float(ix, gpbar, 1, n);
		gsl_vector_float *u1 = gsl_vector_float_alloc(gpbar->size);
		gsl_vector_float *u2 = gsl_vector_float_alloc(gpbar->size);
		orthogonalize32(&src_view.vector, gpbar, u1, u2);
		if (stationary)
			dst = u2->data;
		else
			dst = u1->data;
		gsl_vector_float_free(u1);
		gsl_vector_float_free(u2);
	} else {
		if (stationary)
			dst = src;
		else
			dst = gpbar->data;
	}

	free(ix);
	gsl_vector_float_free(npbar);
	gsl_vector_float_free(gpbar);

	gst_buffer_unmap(inbuf, &in_info);
	gst_buffer_unmap(outbuf, &out_info);
	return GST_FLOW_OK;
}


static GstFlowReturn denoise64(GstBuffer *inbuf, GstBuffer *outbuf, gboolean stationary, gdouble threshold) {

	GstMapInfo in_info, out_info;
	const double *src;
	double *dst;
	int i;
	size_t n = sizeof(src) / sizeof(double);
	size_t *ix[n];
	gsl_vector *npbar = gsl_vector_alloc(n);
	gsl_vector *gpbar = gsl_vector_alloc(n);
	gsl_vector_view src_view = gsl_vector_view_array(src, n);

	gst_buffer_map(inbuf, &in_info, GST_MAP_READ);
	src = (const double *) in_info.data;
	gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE);
	dst = (double *) out_info.data;

	// switch basis to be ordered by magnitude of data
	gsl_sort_index(ix, src, 1, n);
	gsl_permute(ix, src, 1, n);

	// calculate the average expected noise in the new basis
	for (i = 1; i < (n+1); i++) {
		gsl_vector_set(npbar, i, i / n);
	}
	gsl_vector_scale(npbar, 2);
	gsl_vector_add_constant(npbar, -1);
	for (i = 0; i < n; i++) {
		gsl_vector_set(npbar, i, erfinv64(gsl_vector_get(npbar, i)));
	}
	gsl_vector_scale(npbar, M_SQRT2);

	// remove average expected noise
	gsl_vector_memcpy(gpbar, &src_view.vector);
	gsl_vector_sub(gpbar, npbar);

	// keep non-stationary noise in up to predetermined threshold
	size_t half = n / 2;
	gsl_vector_view left = gsl_vector_subvector(gpbar, 0, half);
	gsl_vector_view right = gsl_vector_subvector(gpbar, half, n - half);
	for (i = 0; i < half; i++) {
		if (gsl_vector_get(&left.vector, i) > -threshold)
			gsl_vector_set(&left.vector, i, 0);
	}
	for (i = 0; i < n - half; i++) {
		if (gsl_vector_get(&right.vector, i) < threshold)
			gsl_vector_set(&right.vector, i, 0);
	}

	// decompose into stationary/non-stationary components
	if (!gsl_vector_isnull(gpbar)) {
		gsl_permute_inverse(ix, gpbar, 1, n);
		gsl_vector *u1 = gsl_vector_alloc(gpbar->size);
		gsl_vector *u2 = gsl_vector_alloc(gpbar->size);
		orthogonalize64(&src_view.vector, gpbar, u1, u2);
		if (stationary)
			dst = u2->data;
		else
			dst = u1->data;
		gsl_vector_free(u1);
		gsl_vector_free(u2);
	} else {
		if (stationary)
			dst = src;
		else
			dst = gpbar->data;
	}

	free(ix);
	gsl_vector_free(npbar);
	gsl_vector_free(gpbar);

	gst_buffer_unmap(inbuf, &in_info);
	gst_buffer_unmap(outbuf, &out_info);
	return GST_FLOW_OK;
}


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
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALDenoiser *element = GSTLAL_DENOISER(trans);
	GstAudioInfo info;

	/*
	 * parse the caps
	 */

	if(!gst_audio_info_from_caps(&info, incaps)) {
		GST_ERROR_OBJECT(element, "unable to parse caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/*
	 * set the denoiser function
	 */

	switch(GST_AUDIO_INFO_WIDTH(&info)) {
	case 32:
		element->denoiser_func = denoise32;
		break;

	case 64:
		element->denoiser_func = denoise64;
		break;

	default:
		g_assert_not_reached();
		break;
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALDenoiser *element = GSTLAL_DENOISER(trans);
	GstFlowReturn result;
	GstMapInfo out_info;

	g_assert(element->denoiser_func != NULL);

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		result = element->denoiser_func(inbuf, outbuf, element->stationary, element->threshold);
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


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALDenoiser *element = GSTLAL_DENOISER(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {

	case ARG_STATIONARY:
		element->stationary = g_value_get_boolean(value);
		break;

	case ARG_THRESHOLD:
		element->threshold = g_value_get_double(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALDenoiser *element = GSTLAL_DENOISER(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {

	case ARG_STATIONARY:
		g_value_set_boolean(value, element->stationary);
		break;

	case ARG_THRESHOLD:
		g_value_set_double(value, element->threshold);
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
	GSTLALDenoiser *element = GSTLAL_DENOISER(object);

	element->denoiser_func = NULL;

	G_OBJECT_CLASS(gstlal_denoiser_parent_class)->finalize(object);
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
		"channels = (int) 1, " \
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


static void gstlal_denoiser_class_init(GSTLALDenoiserClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Denoiser",
		"Filter/Audio",
		"Separate out stationary/non-stationary components from signals.",
		"Patrick Godwin <patrick.godwin@ligo.org>, Chad Hanna <chad.hanna@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	properties[ARG_STATIONARY] = g_param_spec_boolean(
		"stationary",
		"Stationary",
		"Return whether to return stationary component (else non-stationary).",
		DEFAULT_STATIONARY,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_THRESHOLD] = g_param_spec_double(
		"threshold",
		"Threshold",
		"The threshold in which to allow non-stationary signals in stationary component",
		0, G_MAXDOUBLE, DEFAULT_THRESHOLD,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);

	g_object_class_install_property(
		gobject_class,
		ARG_STATIONARY,
		properties[ARG_STATIONARY]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_THRESHOLD,
		properties[ARG_THRESHOLD]
	);

}


/*
 * init()
 */


static void gstlal_denoiser_init(GSTLALDenoiser *element)
{
	element->denoiser_func = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	element->stationary = DEFAULT_STATIONARY;
	element->threshold = DEFAULT_THRESHOLD;
}
