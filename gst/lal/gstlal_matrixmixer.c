/*
 * A many-to-many mixer.
 *
 * Copyright (C) 2008  Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna
 * <chad.hanna@ligo.org>
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
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_matrixmixer.h>


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int num_input_channels(const GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		return element->mixmatrix.as_float->size1;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return element->mixmatrix.as_double->size1;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return element->mixmatrix.as_complex_float->size1;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return element->mixmatrix.as_complex_double->size1;

	default:
		g_assert_not_reached();
		return -1;
	}
}


static int num_output_channels(const GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		return element->mixmatrix.as_float->size2;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return element->mixmatrix.as_double->size2;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return element->mixmatrix.as_complex_float->size2;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return element->mixmatrix.as_complex_double->size2;

	default:
		g_assert_not_reached();
		return -1;
	}
}


static size_t mixmatrix_element_size(const GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		return sizeof(*element->mixmatrix.as_float->data);

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return sizeof(*element->mixmatrix.as_double->data);

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return sizeof(*element->mixmatrix.as_complex_float->data);

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return sizeof(*element->mixmatrix.as_complex_double->data);

	default:
		return 0;
	}
}


static void mixmatrix_free(GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		gsl_matrix_float_free(element->mixmatrix.as_float);
		break;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		gsl_matrix_free(element->mixmatrix.as_double);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		gsl_matrix_complex_float_free(element->mixmatrix.as_complex_float);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		gsl_matrix_complex_free(element->mixmatrix.as_complex_double);
		break;

	default:
		break;
	}
}


static GstFlowReturn mix(GSTLALMatrixMixer *element, GstBuffer *inbuf, GstBuffer *outbuf)
{
	guint64 length;
	union {
		gsl_matrix_float_view as_float;
		gsl_matrix_view as_double;
		gsl_matrix_complex_float_view as_complex_float;
		gsl_matrix_complex_view as_complex_double;
	} input_channels, output_channels;

	/*
	 * Wrap the input and output buffers in GSL matrix views, then mix
	 * input channels into output channels.
	 */

	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		input_channels.as_float = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		output_channels.as_float = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element));
		if(input_channels.as_float.matrix.size1 * input_channels.as_float.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.as_float.matrix, element->mixmatrix.as_float, 0, &output_channels.as_float.matrix);
		break;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		input_channels.as_double = gsl_matrix_view_array((double *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		output_channels.as_double = gsl_matrix_view_array((double *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element));
		if(input_channels.as_double.matrix.size1 * input_channels.as_double.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.as_double.matrix, element->mixmatrix.as_double, 0, &output_channels.as_double.matrix);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		input_channels.as_complex_float = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		output_channels.as_complex_float = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element));
		if(input_channels.as_complex_float.matrix.size1 * input_channels.as_complex_float.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_cgemm(CblasNoTrans, CblasNoTrans, (gsl_complex_float) {{1,0}}, &input_channels.as_complex_float.matrix, element->mixmatrix.as_complex_float, (gsl_complex_float) {{0,0}}, &output_channels.as_complex_float.matrix);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		input_channels.as_complex_double = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		output_channels.as_complex_double = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element));
		if(input_channels.as_complex_double.matrix.size1 * input_channels.as_complex_double.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, &input_channels.as_complex_double.matrix, element->mixmatrix.as_complex_double, GSL_COMPLEX_ZERO, &output_channels.as_complex_double.matrix);
		break;
	}

	/*
	 * Done
	 */

	return GST_FLOW_OK;
}


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
		"rate = (int) [ 1, MAX ], " \
		"channels = (int) [ 1, MAX ], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64} ; " \
		"audio/x-raw-complex, " \
		"rate = (int) [ 1, MAX ], " \
		"channels = (int) [ 1, MAX ], " \
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
		"rate = (int) [ 1, MAX ], " \
		"channels = (int) [ 1, MAX ], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64} ; " \
		"audio/x-raw-complex, " \
		"rate = (int) [ 1, MAX ], " \
		"channels = (int) [ 1, MAX ], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


GST_BOILERPLATE(
	GSTLALMatrixMixer,
	gstlal_matrixmixer,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_MATRIX = 1
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
	gint width;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = channels * width / 8;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it can have any number of channels or, if the mixing
		 * matrix is known, the number of channels must equal the
		 * number of rows in the matrix
		 */

		g_mutex_lock(element->mixmatrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->mixmatrix.as_void)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, num_input_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(element->mixmatrix_lock);
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any number of channels or, if the mixing
		 * matrix is known, the number of channels must equal the
		 * number of columns in the matrix
		 */

		g_mutex_lock(element->mixmatrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++)
			if(element->mixmatrix.as_void)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, num_output_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		g_mutex_unlock(element->mixmatrix_lock);
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
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(trans);
	GstStructure *s;
	const char *media_type;
	guint data_type;
	gint in_channels;
	gint out_channels;
	gint width;

	s = gst_caps_get_structure(incaps, 0);
	media_type = gst_structure_get_name(s);
	if(!gst_structure_get_int(s, "channels", &in_channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(s, "width", &width)) {
		GST_DEBUG_OBJECT(element, "unable to parse width from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "channels", &out_channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}

	if(!strcmp(media_type, "audio/x-raw-float")) {
		switch(width) {
		case 32:
			data_type = GSTLAL_MATRIXMIXER_FLOAT;
			break;
		case 64:
			data_type = GSTLAL_MATRIXMIXER_DOUBLE;
			break;
		default:
			return FALSE;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		switch(width) {
		case 64:
			data_type = GSTLAL_MATRIXMIXER_COMPLEX_FLOAT;
			break;
		case 128:
			data_type = GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE;
			break;
		default:
			return FALSE;
		}
	} else
		return FALSE;


	if(!element->mixmatrix.as_void)
		element->data_type = data_type;
	else if(in_channels != num_input_channels(element) || out_channels != num_output_channels(element) || data_type != element->data_type) {
		GST_DEBUG_OBJECT(element, "caps %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT " not accepted:  incorrect data type or wrong channel counts", incaps, outcaps);
		return FALSE;
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(trans);
	GstFlowReturn result;

	g_mutex_lock(element->mixmatrix_lock);
	while(!element->mixmatrix.as_void)
		/* FIXME:  figure out how to get out of this loop */
		g_cond_wait(element->mixmatrix_available, element->mixmatrix_lock);

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		result = mix(element, inbuf, outbuf);
	} else {
		/*
		 * input is 0s.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		result = GST_FLOW_OK;
	}
	g_mutex_unlock(element->mixmatrix_lock);

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
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MATRIX: {
		guint data_type;
		gint in_channels, out_channels;
		g_mutex_lock(element->mixmatrix_lock);
		if(element->mixmatrix.as_void) {
			in_channels = num_input_channels(element);
			out_channels = num_output_channels(element);
			data_type = element->data_type;
			mixmatrix_free(element);
		} else {
			in_channels = out_channels = 0;
			data_type = -1;
		}
		/* FIXME:  allow different data types */
		element->data_type = GSTLAL_MATRIXMIXER_DOUBLE;
		element->mixmatrix.as_double = gstlal_gsl_matrix_from_g_value_array(g_value_get_boxed(value));

		/*
		 * if the data format or number of channels has changed,
		 * force a caps renegotiation
		 */

		if(data_type != element->data_type || num_input_channels(element) != in_channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SINK_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}
		if(data_type != element->data_type || num_output_channels(element) != out_channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}

		g_cond_broadcast(element->mixmatrix_available);
		g_mutex_unlock(element->mixmatrix_lock);
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
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MATRIX:
		g_mutex_lock(element->mixmatrix_lock);
		if(element->mixmatrix.as_void)
			/* FIXME:  allow other data types */
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->mixmatrix.as_double));
		/* FIXME:  else? */
		g_mutex_unlock(element->mixmatrix_lock);
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
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	g_mutex_free(element->mixmatrix_lock);
	element->mixmatrix_lock = NULL;
	g_cond_free(element->mixmatrix_available);
	element->mixmatrix_available = NULL;
	if(element->mixmatrix.as_void) {
		mixmatrix_free(element);
		element->mixmatrix.as_void = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_matrixmixer_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Matrix Mixer", "Filter/Audio", "A many-to-many mixer", "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.org>");

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


static void gstlal_matrixmixer_class_init(GSTLALMatrixMixerClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_MATRIX,
		g_param_spec_value_array(
			"matrix",
			"Matrix",
			"Matrix of mixing coefficients.  Number of rows in matrix sets number of input channels, number of columns sets number of output channels.",
			g_param_spec_value_array(
				"coefficients",
				"Coefficients",
				"Coefficients.",
				/* FIXME:  allow other types */
				g_param_spec_double(
					"coefficient",
					"Coefficient",
					"Coefficient",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_matrixmixer_init(GSTLALMatrixMixer *filter, GSTLALMatrixMixerClass *kclass)
{
	filter->data_type = -1;
	filter->mixmatrix_lock = g_mutex_new();
	filter->mixmatrix_available = g_cond_new();
	filter->mixmatrix.as_void = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}
