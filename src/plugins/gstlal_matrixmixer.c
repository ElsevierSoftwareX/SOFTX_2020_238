/*
 * A many-to-many mixer.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


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
		return element->mixmatrix.as_float.matrix.size1;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return element->mixmatrix.as_double.matrix.size1;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return element->mixmatrix.as_complex_float.matrix.size1;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return element->mixmatrix.as_complex_double.matrix.size1;

	default:
		return 0;
	}
}


static int num_output_channels(const GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		return element->mixmatrix.as_float.matrix.size2;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return element->mixmatrix.as_double.matrix.size2;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return element->mixmatrix.as_complex_float.matrix.size2;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return element->mixmatrix.as_complex_double.matrix.size2;

	default:
		return 0;
	}
}


static size_t mixmatrix_element_size(const GSTLALMatrixMixer *element)
{
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		return sizeof(*element->mixmatrix.as_float.matrix.data);

	case GSTLAL_MATRIXMIXER_DOUBLE:
		return sizeof(*element->mixmatrix.as_double.matrix.data);

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		return sizeof(*element->mixmatrix.as_complex_float.matrix.data);

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		return sizeof(*element->mixmatrix.as_complex_double.matrix.data);

	default:
		return 0;
	}
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


static GstCaps *getcaps(GstPad *pad)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	GstCaps *caps;
	GstCaps *peercaps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);
	GST_DEBUG_OBJECT(element, "our caps = %" GST_PTR_FORMAT, caps);

	/*
	 * now compute the intersection of the caps with the downstream
	 * peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result;
		guint n;
		for(n = 0; n < gst_caps_get_size(peercaps); n++)
			gst_structure_remove_field(gst_caps_get_structure(peercaps, n), "channels");
		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " with %" GST_PTR_FORMAT, caps, peercaps);
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection = %" GST_PTR_FORMAT, caps);
	}

	/*
	 * if we have a mixing matrix the sink pad's media type and sample
	 * width must be the same as the mixing matrix's, and the number of
	 * channels must match the number of rows in the mixing matrix.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(element->mixmatrix_buf) {
		GstCaps *matrixcaps = gst_caps_make_writable(gst_buffer_get_caps(element->mixmatrix_buf));
		GstCaps *result;
		guint n;

		for(n = 0; n < gst_caps_get_size(matrixcaps); n++)
			gst_structure_set(gst_caps_get_structure(matrixcaps, n), "channels", G_TYPE_INT, num_input_channels(element), NULL);
		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " with %" GST_PTR_FORMAT, caps, matrixcaps);
		result = gst_caps_intersect(matrixcaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(matrixcaps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection = %" GST_PTR_FORMAT, caps);
	}
	g_mutex_unlock(element->mixmatrix_lock);

	/*
	 * done.
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	/*
	 * if we have a mixing matrix the caps' media type and sample width
	 * must be the same as the mixing matrix's, and the number of
	 * channels must match the number of columns in the mixing matrix.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(success && element->mixmatrix_buf) {
		GstCaps *matrixcaps = gst_caps_make_writable(gst_buffer_get_caps(element->mixmatrix_buf));
		GstCaps *result;
		guint n;

		for(n = 0; n < gst_caps_get_size(matrixcaps); n++)
			gst_structure_set(gst_caps_get_structure(matrixcaps, n), "channels", G_TYPE_INT, num_input_channels(element), NULL);
		result = gst_caps_intersect(matrixcaps, caps);
		success = !gst_caps_is_empty(result);
		gst_caps_unref(matrixcaps);
		gst_caps_unref(result);

		/*
		 * use the mixing matrix to set the number of output
		 * channels to the number of columns in the mixing matrix
		 * and check if the downstream peer will accept the caps.
		 * gst_caps_make_writable() unref()s its argument so we
		 * have to ref() it first to keep it valid.
		 */

		if(success) {
			gst_caps_ref(caps);
			caps = gst_caps_make_writable(caps);

			for(n = 0; n < gst_caps_get_size(caps); n++)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, num_output_channels(element), NULL);
			GST_DEBUG_OBJECT(element, "trying to set " GST_PTR_FORMAT " on downstream peer", caps);
			success = gst_pad_set_caps(element->srcpad, caps);
			gst_caps_unref(caps);
		}
	}
	g_mutex_unlock(element->mixmatrix_lock);

	/*
	 * done.
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	guint64 samples;
	union {
		gsl_matrix_float_view as_float;
		gsl_matrix_view as_double;
		gsl_matrix_complex_float_view as_complex_float;
		gsl_matrix_complex_view as_complex_double;
	} input_channels;
	GstBuffer *srcbuf;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * Wrap the incoming buffer in a GSL matrix view.
	 */

	if(!(GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf))) {
		GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer has invalid offset and/or end ofset", sinkbuf));
		result = GST_FLOW_ERROR;
		goto done;
	}
	samples = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		input_channels.as_float = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(sinkbuf), samples, num_input_channels(element));
		if(input_channels.as_float.matrix.size1 * input_channels.as_float.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", sinkbuf));
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		break;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		input_channels.as_double = gsl_matrix_view_array((double *) GST_BUFFER_DATA(sinkbuf), samples, num_input_channels(element));
		if(input_channels.as_double.matrix.size1 * input_channels.as_double.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", sinkbuf));
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		input_channels.as_complex_float = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(sinkbuf), samples, num_input_channels(element));
		if(input_channels.as_complex_float.matrix.size1 * input_channels.as_complex_float.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", sinkbuf));
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		input_channels.as_complex_double = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(sinkbuf), samples, num_input_channels(element));
		if(input_channels.as_complex_double.matrix.size1 * input_channels.as_complex_double.matrix.size2 * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("%p: buffer size is not an integer number of samples", sinkbuf));
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		break;
	}

	/*
	 * Make sure we have a mixing matrix, wait until we do.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(!element->mixmatrix_buf) {
		g_cond_wait(element->mixmatrix_available, element->mixmatrix_lock);
		if(!element->mixmatrix_buf) {
			g_mutex_unlock(element->mixmatrix_lock);
			/* mixing matrix didn't get set.  probably means
			 * we're being disposed(). */
			GST_ELEMENT_ERROR(element, STREAM, FAILED, (NULL), ("mixing matrix not available"));
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
	}

	/*
	 * Get a buffer from the downstream peer, and copy the metadata
	 * from the input buffer.
	 */

	result = gst_pad_alloc_buffer(element->srcpad, GST_BUFFER_OFFSET(sinkbuf), samples * num_output_channels(element) * mixmatrix_element_size(element), GST_PAD_CAPS(element->srcpad), &srcbuf);
	if(result != GST_FLOW_OK) {
		g_mutex_unlock(element->mixmatrix_lock);
		GST_ELEMENT_ERROR(element, CORE, PAD, (NULL), ("%s: gst_pad_alloc_buffer() failed (%d)", GST_PAD_NAME(element->srcpad), result));
		goto done;
	}
	gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);

	/*
	 * Math.  Just memset() the output to 0 if the input buffer is a
	 * gap.
	 */

	if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP))
		memset(GST_BUFFER_DATA(srcbuf), 0, GST_BUFFER_SIZE(srcbuf));
	else {
		union {
			gsl_matrix_float_view as_float;
			gsl_matrix_view as_double;
			gsl_matrix_complex_float_view as_complex_float;
			gsl_matrix_complex_view as_complex_double;
		} output_channels;

		/*
		 * Wrap the outgoing buffer in a GSL matrix view, then mix
		 * input channels into output channels.
		 */

		switch(element->data_type) {
		case GSTLAL_MATRIXMIXER_FLOAT:
			output_channels.as_float = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(srcbuf), samples, num_output_channels(element));
			gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.as_float.matrix, &element->mixmatrix.as_float.matrix, 0, &output_channels.as_float.matrix);
			break;

		case GSTLAL_MATRIXMIXER_DOUBLE:
			output_channels.as_double = gsl_matrix_view_array((double *) GST_BUFFER_DATA(srcbuf), samples, num_output_channels(element));
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.as_double.matrix, &element->mixmatrix.as_double.matrix, 0, &output_channels.as_double.matrix);
			break;

		case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
			output_channels.as_complex_float = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(srcbuf), samples, num_output_channels(element));
			/* FIXME:  frigging GSL decides it needs its own
			 * *special* complex type, then doesn't provide a
			 * complete suite of support functions.  now we
			 * have to create 1 and 0 by hand */
			gsl_blas_cgemm(CblasNoTrans, CblasNoTrans, (gsl_complex_float) {{1,0}}, &input_channels.as_complex_float.matrix, &element->mixmatrix.as_complex_float.matrix, (gsl_complex_float) {{0,0}}, &output_channels.as_complex_float.matrix);
			break;

		case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
			output_channels.as_complex_double = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(srcbuf), samples, num_output_channels(element));
			gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, &input_channels.as_complex_double.matrix, &element->mixmatrix.as_complex_double.matrix, GSL_COMPLEX_ZERO, &output_channels.as_complex_double.matrix);
			break;
		}
	}
	g_mutex_unlock(element->mixmatrix_lock);

	/*
	 * Push the buffer downstream
	 */

	result = gst_pad_push(element->srcpad, srcbuf);
	if(result != GST_FLOW_OK) {
		GST_ELEMENT_ERROR(element, CORE, PAD, (NULL), ("%s: gst_pad_push() failed (%d)", GST_PAD_NAME(element->srcpad), result));
		goto done;
	}

	/*
	 * Done
	 */

done:
	gst_buffer_unref(sinkbuf);
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                                 Matrix Pad
 *
 * ============================================================================
 */


/*
 * setcaps()
 */


static gboolean setcaps_matrix(GstPad *pad, GstCaps *caps)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	GstStructure *structure = gst_caps_get_structure(caps, 0);
	const char *media_type;
	gint width;
	gboolean success = TRUE;

	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "width", &width);

	if(!strcmp(media_type, "audio/x-raw-float")) {
		switch(width) {
		case 32:
			element->data_type = GSTLAL_MATRIXMIXER_FLOAT;
			break;
		case 64:
			element->data_type = GSTLAL_MATRIXMIXER_DOUBLE;
			break;
		default:
			success = FALSE;
			break;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		switch(width) {
		case 64:
			element->data_type = GSTLAL_MATRIXMIXER_COMPLEX_FLOAT;
			break;
		case 128:
			element->data_type = GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE;
			break;
		default:
			success = FALSE;
			break;
		}
	} else
		success = FALSE;

	/*
	 * done.
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain_matrix(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstStructure *structure = gst_caps_get_structure(caps, 0);
	GstFlowReturn result = GST_FLOW_OK;
	int rows;
	int cols;

	/*
	 * Get the matrix size.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	gst_structure_get_int(structure, "channels", &cols);
	rows = GST_BUFFER_SIZE(sinkbuf) / mixmatrix_element_size(element) / cols;
	if(rows * cols * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
		GST_ERROR_OBJECT(element, "buffer size mismatch:  input buffer size not divisible by the channel count");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Replace the current matrix with the new one.
	 */

	if(element->mixmatrix_buf)
		gst_buffer_unref(element->mixmatrix_buf);
	element->mixmatrix_buf = sinkbuf;
	switch(element->data_type) {
	case GSTLAL_MATRIXMIXER_FLOAT:
		element->mixmatrix.as_float = gsl_matrix_float_view_array((float *) GST_BUFFER_DATA(sinkbuf), rows, cols);
		break;

	case GSTLAL_MATRIXMIXER_DOUBLE:
		element->mixmatrix.as_double = gsl_matrix_view_array((double *) GST_BUFFER_DATA(sinkbuf), rows, cols);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_FLOAT:
		element->mixmatrix.as_complex_float = gsl_matrix_complex_float_view_array((float *) GST_BUFFER_DATA(sinkbuf), rows, cols);
		break;

	case GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE:
		element->mixmatrix.as_complex_double = gsl_matrix_complex_view_array((double *) GST_BUFFER_DATA(sinkbuf), rows, cols);
		break;
	}
	g_cond_signal(element->mixmatrix_available);

	/*
	 * Force a renegotiation of the format on the sink pad
	 */

	gst_pad_set_caps(element->sinkpad, NULL);

	/*
	 * Done
	 */

done:
	g_mutex_unlock(element->mixmatrix_lock);
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	gst_object_unref(element->matrixpad);
	element->matrixpad = NULL;
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_mutex_free(element->mixmatrix_lock);
	element->mixmatrix_lock = NULL;
	g_cond_free(element->mixmatrix_available);
	element->mixmatrix_available = NULL;
	if(element->mixmatrix_buf) {
		gst_buffer_unref(element->mixmatrix_buf);
		element->mixmatrix_buf = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Matrix Mixer",
		"Filter",
		"A many-to-many mixer",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"matrix",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} ; " \
				"audio/x-raw-complex, " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
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
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
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
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->finalize = finalize;
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) matrix pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "matrix");
	gst_pad_set_setcaps_function(pad, setcaps_matrix);
	gst_pad_set_chain_function(pad, chain_matrix);
	element->matrixpad = pad;

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_getcaps_function(pad, getcaps);
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->mixmatrix_lock = g_mutex_new();
	element->mixmatrix_available = g_cond_new();
	element->mixmatrix_buf = NULL;
}


/*
 * gstlal_matrixmixer_get_type().
 */


GType gstlal_matrixmixer_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALMatrixMixerClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALMatrixMixer),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_matrixmixer", &info, 0);
	}

	return type;
}
