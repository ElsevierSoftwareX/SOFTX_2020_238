/*
 * A many-to-many mixer.
 *
 * Copyright (C) 2008--2013  Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna
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


/**
 * SECTION:gstlal_matrixmixer
 * @short_description:  Compute linear combinations of time series channels.
 *
 * Generates one or more time series channels as linear combinations of one
 * or more input channels.  The mixer is controlled using a matrix whose
 * elements provide the mixing coefficients and whose size sets the number
 * of input and output channels.  The meaning of the coefficients is shown
 * in the following diagram.
 *
 * <informalequation>
 * 	<xi:include href="../math/eqn_matrixmixer_01.mml"/>
 * </informalequation>
 *
 * The matrix is provided to the element via the "matrix" property as a
 * GValueArray of rows, each row is a GValueArray containing
 * double-precision floats (the rows must all be the same size).
 *
 * The coefficient ordering and the manner in which they map input channels
 * to output channels is chosen so that the transformation of an input
 * buffer into an output buffer can be performed as a single matrix-matrix
 * multiplication, to allow optimized platform-specific implementations of
 * that operation to be employed.
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
#include <gst/controller/gstcontroller.h>


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


#include <gstlal/gstlal.h>
#include <gstlal_matrixmixer.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_matrixmixer_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_matrixmixer", 0, "lal_matrixmixer element");
}


GST_BOILERPLATE_FULL(
	GSTLALMatrixMixer,
	gstlal_matrixmixer,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static enum gstlal_matrixmixer_media_type get_media_type(GstCaps *caps)
{
	GstStructure *s = gst_caps_get_structure(caps, 0);
	const char *media_type = gst_structure_get_name(s);
	gint width;

	if(!gst_structure_get_int(s, "width", &width))
		return GSTLAL_MATRIXMIXER_NONE;

	if(!strcmp(media_type, "audio/x-raw-float")) {
		switch(width) {
		case 32:
			return GSTLAL_MATRIXMIXER_FLOAT;
		case 64:
			return GSTLAL_MATRIXMIXER_DOUBLE;
		default:
			return GSTLAL_MATRIXMIXER_NONE;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		switch(width) {
		case 64:
			return GSTLAL_MATRIXMIXER_COMPLEX_FLOAT;
		case 128:
			return GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE;
		default:
			return GSTLAL_MATRIXMIXER_NONE;
		}
	} else
		return GSTLAL_MATRIXMIXER_NONE;
}


static int num_input_channels(const GSTLALMatrixMixer *element)
{
	return element->mixmatrix_d->size1;
}


static int num_output_channels(const GSTLALMatrixMixer *element, enum gstlal_matrixmixer_media_type data_type)
{
	switch(data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
	case GSTLAL_MATRIXMIXER_DOUBLE:
		return element->mixmatrix_d->size2;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return element->mixmatrix_d->size2 / 2;

	default:
		g_assert_not_reached();
		return -1;
	}
}


static void mixmatrix_free(GSTLALMatrixMixer *element)
{
	if(element->mixmatrix_d) {
		gsl_matrix_free(element->mixmatrix_d);
		element->mixmatrix_d = NULL;
	}
	if(element->mixmatrix_s) {
		gsl_matrix_float_free(element->mixmatrix_s);
		element->mixmatrix_s = NULL;
	}
}


static GstFlowReturn mix(GSTLALMatrixMixer *element, GstBuffer *inbuf, GstBuffer *outbuf)
{
	guint64 length;

	/*
	 * Number of samples to process.
	 */

	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(!length) {
		g_assert_cmpuint(GST_BUFFER_SIZE(inbuf), ==, 0);
		return GST_FLOW_OK;
	}
	g_assert_cmpuint(GST_BUFFER_SIZE(inbuf) % length, ==, 0);

	/*
	 * Wrap the input and output buffers in GSL matrix views, then mix
	 * input channels into output channels.
	 */

	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT: {
		gsl_matrix_float_view input_channels = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		gsl_matrix_float_view output_channels = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element, element->data_type));
		if(input_channels.matrix.size1 * input_channels.matrix.size2 * sizeof(float) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size does not match channel and sample count", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.matrix, element->mixmatrix_s, 0, &output_channels.matrix);
		break;
	}

	case GSTLAL_MATRIXMIXER_DOUBLE: {
		gsl_matrix_view input_channels = gsl_matrix_view_array((double *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		gsl_matrix_view output_channels = gsl_matrix_view_array((double *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element, element->data_type));
		if(input_channels.matrix.size1 * input_channels.matrix.size2 * sizeof(double) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size does not match channel and sample count", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.matrix, element->mixmatrix_d, 0, &output_channels.matrix);
		break;
	}

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT: {
		gsl_matrix_complex_float_view input_channels = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		gsl_matrix_complex_float_view output_channels = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element, element->data_type));
		if(input_channels.matrix.size1 * input_channels.matrix.size2 * sizeof(complex float) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size does not match channel and sample count", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_cgemm(CblasNoTrans, CblasNoTrans, (gsl_complex_float) {{1,0}}, &input_channels.matrix, &element->mixmatrix_cs.matrix, (gsl_complex_float) {{0,0}}, &output_channels.matrix);
		break;
	}

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE: {
		gsl_matrix_complex_view input_channels = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(inbuf), length, num_input_channels(element));
		gsl_matrix_complex_view output_channels = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(outbuf), length, num_output_channels(element, element->data_type));
		if(input_channels.matrix.size1 * input_channels.matrix.size2 * sizeof(complex double) != GST_BUFFER_SIZE(inbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size does not match channel and sample count", inbuf));
			return GST_FLOW_NOT_NEGOTIATED;
		}
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, &input_channels.matrix, &element->mixmatrix_cd.matrix, GSL_COMPLEX_ZERO, &output_channels.matrix);
		break;
	}

	default:
		g_assert_not_reached();
	}

	/*
	 * Done
	 */

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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint width;
	gint channels;
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);
	success &= gst_structure_get_int(str, "width", &width);

	if(success)
		*size = channels * width / 8;
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(trans);
	enum gstlal_matrixmixer_media_type data_type = get_media_type(caps);
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
			if(element->mixmatrix_d)
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
			if(element->mixmatrix_d)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, num_output_channels(element, data_type), NULL);
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
	enum gstlal_matrixmixer_media_type data_type = get_media_type(incaps);
	gint in_channels;
	gint out_channels;
	gboolean success = data_type != GSTLAL_MATRIXMIXER_NONE;

	success &= gst_structure_get_int(gst_caps_get_structure(incaps, 0), "channels", &in_channels);
	success &= gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "channels", &out_channels);

	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse incaps %" GST_PTR_FORMAT ", outcaps %" GST_PTR_FORMAT, incaps, outcaps);
	else {
		guint old_datatype = element->data_type;

		g_mutex_lock(element->mixmatrix_lock);
		element->data_type = data_type;
		if(element->mixmatrix_d && (in_channels != num_input_channels(element) || out_channels != num_output_channels(element, element->data_type))) {
			GST_WARNING_OBJECT(element, "caps %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT " not accepted, wrong channel counts:  (%d in, %d out) != (%d in, %d out)", incaps, outcaps, in_channels, out_channels, num_input_channels(element), num_output_channels(element, element->data_type));
			element->data_type = old_datatype;
			success = FALSE;
		}
		g_mutex_unlock(element->mixmatrix_lock);
	}

	return success;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(trans);
	GstFlowReturn result;

	gst_object_sync_values(G_OBJECT(trans), GST_BUFFER_TIMESTAMP(inbuf));

	g_mutex_lock(element->mixmatrix_lock);
	while(!element->mixmatrix_d) {
		GST_DEBUG_OBJECT(element, "mix matrix not available, waiting ...");
		g_cond_wait(element->mixmatrix_available, element->mixmatrix_lock);
		if(GST_STATE(GST_ELEMENT(trans)) == GST_STATE_NULL) {
			GST_DEBUG_OBJECT(element, "element now in null state, abandoning wait for mix matrix");
			result = GST_FLOW_WRONG_STATE;
			goto done;
		}
	}

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
		/* prepare_output_buffer() lied.  tell the truth */
		/* FIXME:  put back when resampler can handle non-malloc()ed buffers */
		/*GST_BUFFER_SIZE(outbuf) = 0;*/
		/* FIXME:  this is needed if used in pipelines that don't
		 * understand gaps at all */
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		result = GST_FLOW_OK;
	}

	/*
	 * done
	 */

done:
	gst_buffer_copy_metadata(outbuf, inbuf, GST_BUFFER_COPY_TIMESTAMPS);
	g_mutex_unlock(element->mixmatrix_lock);
	return result;
}


/*
 * prepare_output_buffer()
 */


static GstFlowReturn prepare_output_buffer(GstBaseTransform *trans, GstBuffer *input, gint size, GstCaps *caps, GstBuffer **buf)
{
	GstFlowReturn result;

	/* FIXME:  put back commented-out code when resampler can handle non-malloc()ed buffers */
	result = gst_pad_alloc_buffer(GST_BASE_TRANSFORM_SRC_PAD(trans), GST_BUFFER_OFFSET(input), /*GST_BUFFER_FLAG_IS_SET(input, GST_BUFFER_FLAG_GAP) ? 0 :*/ size, caps, buf);
	if(result != GST_FLOW_OK)
		goto done;

	/* lie to trick basetransform */
	GST_BUFFER_SIZE(*buf) = size;

done:
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
	ARG_MATRIX = 1
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MATRIX: {
		gint in_channels = 0;
		gint out_channels = 0;
		g_mutex_lock(element->mixmatrix_lock);
		if(element->mixmatrix_d) {
			if(element->data_type) {
				in_channels = num_input_channels(element);
				out_channels = num_output_channels(element, element->data_type);
			}
			mixmatrix_free(element);
		}
		element->mixmatrix_d = gstlal_gsl_matrix_from_g_value_array(g_value_get_boxed(value));

		/*
		 * copy to different types
		 */

		element->mixmatrix_s = gsl_matrix_float_alloc(element->mixmatrix_d->size1, element->mixmatrix_d->size2);
		{
		guint i, j;
		for(i = 0; i < element->mixmatrix_d->size1; i++) for(j = 0; j < element->mixmatrix_d->size2; j++)
			gsl_matrix_float_set(element->mixmatrix_s, i, j, gsl_matrix_get(element->mixmatrix_d, i, j));
		}
		if(element->mixmatrix_d->size2 >= 2) {
			element->mixmatrix_cd = gsl_matrix_complex_view_array(element->mixmatrix_d->data, element->mixmatrix_d->size1, element->mixmatrix_d->size2 / 2);
			element->mixmatrix_cs = gsl_matrix_complex_float_view_array(element->mixmatrix_s->data, element->mixmatrix_s->size1, element->mixmatrix_s->size2 / 2);
		}

		/*
		 * if the number of channels has changed, force a caps
		 * renegotiation
		 */

		if(num_input_channels(element) != in_channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SINK_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}
		if(element->data_type && num_output_channels(element, element->data_type) != out_channels) {
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
		if(element->mixmatrix_d)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->mixmatrix_d));
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
 * dispose()
 */


static void dispose(GObject *object)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	/*
	 * wake up any threads that are waiting for the mix matrix to
	 * become available;  since we are being finalized the element
	 * state should be NULL causing those threads to bail out
	 */

	g_mutex_lock(element->mixmatrix_lock);
	g_cond_broadcast(element->mixmatrix_available);
	g_mutex_unlock(element->mixmatrix_lock);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	/*
	 * free resources
	 */

	g_mutex_free(element->mixmatrix_lock);
	element->mixmatrix_lock = NULL;
	g_cond_free(element->mixmatrix_available);
	element->mixmatrix_available = NULL;
	mixmatrix_free(element);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_matrixmixer_base_init(gpointer klass)
{
}


/*
 * class_init()
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


static void gstlal_matrixmixer_class_init(GSTLALMatrixMixerClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(element_class, "Matrix Mixer", "Filter/Audio", "A many-to-many mixer", "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.org>");

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(dispose);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
		)
	);
}


/*
 * init()
 */


static void gstlal_matrixmixer_init(GSTLALMatrixMixer *filter, GSTLALMatrixMixerClass *klass)
{
	filter->data_type = GSTLAL_MATRIXMIXER_NONE;
	filter->mixmatrix_lock = g_mutex_new();
	filter->mixmatrix_available = g_cond_new();
	filter->mixmatrix_d = NULL;
	filter->mixmatrix_s = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}
