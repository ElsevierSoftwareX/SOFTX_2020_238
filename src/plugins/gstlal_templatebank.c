/*
 * A template bank.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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


#include <math.h>
#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from LAL
 */


/*
 * stuff from GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_templatebank.h>
#include <low_latency_inspiral_functions.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_T_START 0
#define DEFAULT_T_END G_MAXDOUBLE
#define DEFAULT_SNR_LENGTH 2048	/* samples */
#define TEMPLATE_SAMPLE_RATE 4096	/* Hertz */
#define TOLERANCE 0.97


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/**
 * Template bank properties:  number of samples in each template.
 */


static int template_length(const GSTLALTemplateBank *element)
{
	return element->U->size2;
}


/**
 * Template bank properties:  number of templates.
 */


static int num_templates(const GSTLALTemplateBank *element)
{
	return element->U->size1;
}


/**
 * Create and destroy the orthonormal basis for the template bank.
 */


static void svd_destroy(GSTLALTemplateBank *element)
{
	if(element->U) {
		gsl_matrix_free(element->U);
		element->U = NULL;
	}
	if(element->S) {
		gsl_vector_free(element->S);
		element->S = NULL;
	}
	if(element->V) {
		gsl_matrix_free(element->V);
		element->V = NULL;
	}
	if(element->chifacs) {
		gsl_vector_free(element->chifacs);
		element->chifacs = NULL;
	}
}


static int svd_create(GSTLALTemplateBank *element, int sample_rate)
{
	int verbose = 1;

	/*
	 * be sure we don't leak memory
	 */

	svd_destroy(element);

	/*
	 * clip t_start and t_end so that 0 <= t_start <= t_end <=
	 * element->t_total_duration (both are unsigned so can't be
	 * negative)
	 */

	if(element->t_start > element->t_total_duration)
		element->t_start = element->t_total_duration;
	if(element->t_end < element->t_start)
		element->t_end = element->t_start;
	else if(element->t_end > element->t_total_duration)
		element->t_end = element->t_total_duration;

	/*
	 * generate orthonormal template bank
	 */

	generate_bank_svd(&element->U, &element->S, &element->V, &element->chifacs, element->template_bank_filename, element->reference_psd_filename, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / sample_rate, element->t_start, element->t_end, element->t_total_duration, TOLERANCE, verbose);

	/*
	 * done
	 */

	return 0;
}


/**
 * Transmit the mixer matrix to the mixer element, downstream in the
 * pipeline.
 */


static GstFlowReturn push_mixer_matrix(GstPad *pad, gsl_matrix *matrix, GstClockTime timestamp)
{
	GstBuffer *buf;
	GstCaps *caps;
	gboolean success;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * Negotiate the matrix size with the mixer.
	 */

	caps = gst_caps_new_simple(
		"audio/x-raw-float",
		"channels", G_TYPE_INT, matrix->size2,
		"endianness", G_TYPE_INT, G_BYTE_ORDER,
		"width", G_TYPE_INT, 64,
		NULL
	);
	success = gst_pad_set_caps(pad, caps);
	gst_caps_unref(caps);
	if(!success) {
		GST_ERROR("failure negotiating caps with mixer");
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Get a buffer from the mixer.
	 */

	result = gst_pad_alloc_buffer(pad, GST_BUFFER_OFFSET_NONE, matrix->size1 * matrix->size2 * sizeof(*matrix->data), GST_PAD_CAPS(pad), &buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("failure getting buffer from mixer");
		goto done;
	}

	/*
	 * Set the metadata.
	 */

	GST_BUFFER_TIMESTAMP(buf) = timestamp;

	/*
	 * Copy the matrix data into the buffer.
	 */

	memcpy(GST_BUFFER_DATA(buf), matrix->data, GST_BUFFER_SIZE(buf));

	/*
	 * Push the buffer downstream.
	 */

	result = gst_pad_push(pad, buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("mixer won't accept matrix");
		goto done;
	}

	/*
	 * Done.
	 */

done:
	return result;
}


/**
 * Normalization correction because we don't know what we're doing.
 *
 * This is a fit to the standard deviations of the low-passed and
 * down-sampled time series.  The measured values were:
 *
 *	16384	1.0129
 *	 4096	0.50173
 *	 2048	0.35129
 *	  512	0.16747
 *	  256	0.11191
 *	  128	0.077322
 *
 * dividing the input time series by this gives it a mean square of 1.
 */


static double normalization_correction(double rate)
{
	return exp(0.53222 * log(rate) - 5.1269);
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_TEMPLATE_BANK = 1,
	ARG_REFERENCE_PSD,
	ARG_T_START,
	ARG_T_END,
	ARG_T_TOTAL_DURATION,
	ARG_SNR_LENGTH
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	switch(id) {
	case ARG_TEMPLATE_BANK:
		free(element->template_bank_filename);
		element->template_bank_filename = g_value_dup_string(value);
		break;

	case ARG_REFERENCE_PSD:
		free(element->reference_psd_filename);
		element->reference_psd_filename = g_value_dup_string(value);
		break;

	case ARG_T_START:
		element->t_start = g_value_get_double(value);
		break;

	case ARG_T_END:
		element->t_end = g_value_get_double(value);
		break;

	case ARG_T_TOTAL_DURATION:
		element->t_total_duration = g_value_get_double(value);
		break;

	case ARG_SNR_LENGTH:
		element->snr_length = g_value_get_uint(value);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	switch(id) {
	case ARG_TEMPLATE_BANK:
		g_value_set_string(value, element->template_bank_filename);
		break;

	case ARG_REFERENCE_PSD:
		g_value_set_string(value, element->reference_psd_filename);
		break;

	case ARG_T_START:
		g_value_set_double(value, element->t_start);
		break;

	case ARG_T_END:
		g_value_set_double(value, element->t_end);
		break;

	case ARG_T_TOTAL_DURATION:
		g_value_set_double(value, element->t_total_duration);
		break;

	case ARG_SNR_LENGTH:
		g_value_set_uint(value, element->snr_length);
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
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	element->sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	result = gst_pad_set_caps(element->sumsquarespad, caps);

	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	int output_length;
	int i;

	/*
	 * Now that we know the sample rate, construct orthogonal basis for
	 * the template bank if not already done.
	 */

	if(!element->U) {
		GstCaps *caps;
		gboolean success;

		/*
		 * Create the orthogonal basis.
		 */

		svd_create(element, element->sample_rate);

		/*
		 * Tell the mixer how to reconstruct the SNRs.
		 */

		result = push_mixer_matrix(element->matrixpad, element->V, GST_BUFFER_TIMESTAMP(sinkbuf));
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Now that we know how many channels we'll produce, set
		 * the srcpad's caps.  gst_caps_make_writable() unref()s
		 * its argument.
		 */

		caps = gst_caps_make_writable(gst_buffer_get_caps(sinkbuf));
		gst_caps_set_simple(caps, "channels", G_TYPE_INT, num_templates(element), NULL);
		success = gst_pad_set_caps(element->srcpad, caps);
		gst_caps_unref(caps);
		if(success != TRUE) {
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
	}

	/*
	 * Check for a discontinuity.
	 */

	if(GST_BUFFER_IS_DISCONT(sinkbuf)) {
		GstBuffer *zeros;

		/*
		 * Remember to set the discontinuity flag on the next
		 * buffer we push out.
		 */

		element->next_is_discontinuity = TRUE;

		/*
		 * Clear the contents of the adpater
		 *
		 * FIXME:  how 'bout pushing in more zeros and finishing
		 * off the contents of the adapter first?
		 */

		gst_adapter_clear(element->adapter);

		/*
		 * Synchronize the output sample counter and time stamp
		 * with the input buffer.
		 */

		if(GST_BUFFER_OFFSET_IS_VALID(sinkbuf))
			element->next_sample = GST_BUFFER_OFFSET(sinkbuf);
		if(GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf))
			element->output_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);

		/*
		 * Push GAP buffers of zeros out both src pads to pad the
		 * output streams.
		 */

		if(element->t_start) {
			result = gst_pad_alloc_buffer(element->sumsquarespad, element->next_sample, (size_t) floor(element->t_start * element->sample_rate + 0.5) * sizeof(*element->U->data), GST_PAD_CAPS(element->sumsquarespad), &zeros);
			if(result != GST_FLOW_OK)
				goto done;
			memset(GST_BUFFER_DATA(zeros), 0, GST_BUFFER_SIZE(zeros));
			gst_buffer_copy_metadata(zeros, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
			GST_BUFFER_FLAG_SET(zeros, GST_BUFFER_FLAG_GAP);
			GST_BUFFER_OFFSET_END(zeros) = element->next_sample + (size_t) floor(element->t_start * element->sample_rate + 0.5);
			GST_BUFFER_DURATION(zeros) = (GstClockTime) floor(element->t_start * GST_SECOND + 0.5);

			result = gst_pad_push(element->sumsquarespad, zeros);
			if(result != GST_FLOW_OK)
				goto done;

			result = gst_pad_alloc_buffer(element->srcpad, element->next_sample, num_templates(element) * (size_t) floor(element->t_start * element->sample_rate + 0.5) * sizeof(*element->U->data), GST_PAD_CAPS(element->srcpad), &zeros);
			if(result != GST_FLOW_OK)
				goto done;
			memset(GST_BUFFER_DATA(zeros), 0, GST_BUFFER_SIZE(zeros));
			gst_buffer_copy_metadata(zeros, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
			GST_BUFFER_FLAG_SET(zeros, GST_BUFFER_FLAG_GAP);
			GST_BUFFER_OFFSET_END(zeros) = element->next_sample + (size_t) floor(element->t_start * element->sample_rate + 0.5);
			GST_BUFFER_DURATION(zeros) = (GstClockTime) floor(element->t_start * GST_SECOND + 0.5);

			result = gst_pad_push(element->srcpad, zeros);
			if(result != GST_FLOW_OK)
				goto done;

			/*
			 * Update the offset and time stamp.
			 */

			element->next_is_discontinuity = FALSE;
			element->next_sample += (size_t) floor(element->t_start * element->sample_rate + 0.5);
			element->output_timestamp += (GstClockTime) floor(element->t_start * GST_SECOND + 0.5);
		}

		/*
		 * Pad the adapter with enough 0s to accomodate the
		 * template minus 1 sample, so that the first SNR sample
		 * generated is for when the first input sample intersects
		 * the start of the template.
		 */

		zeros = gst_buffer_try_new_and_alloc((template_length(element) - 1) * sizeof(*element->U->data));
		if(!zeros) {
			result = GST_FLOW_ERROR;
			goto done;
		}
		memset(GST_BUFFER_DATA(zeros), 0, GST_BUFFER_SIZE(zeros));
		gst_adapter_push(element->adapter, zeros);
	}

	/*
	 * Put buffer into adapter.
	 */

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * While there is enough data in the adapter to produce a buffer of
	 * SNR.
	 */

	while(1) {
		GstBuffer *orthogonal_snr_buf;
		GstBuffer *orthogonal_snr_sum_squares_buf;
		gsl_vector_view time_series;
		gsl_matrix_view orthogonal_snr;
		gsl_vector *orthogonal_snr_sample;
		gsl_vector_view orthogonal_snr_sum_squares;

		/*
		 * How many SNR samples can we construct from the contents
		 * of the adapter?  The +1 is because when there is 1
		 * template-length of data in the adapter then we can
		 * produce 1 SNR sample, not 0.
		 */

		output_length = (int) (gst_adapter_available(element->adapter) / sizeof(*time_series.vector.data)) - template_length(element) + 1;

		/*
		 * Clip to the requested output length.  Quit the loop if
		 * there aren't enough samples.
		 */

		if(element->snr_length) {
			if(output_length < (int) element->snr_length)
				break;
			output_length = element->snr_length;
		} else if(output_length <= 0)
			break;

		/*
		 * Wrap the adapter's contents in a GSL vector view.  To
		 * produce output_length SNR samples requires output_length
		 * + template_length - 1 samples from the adapter.  Note
		 * that the wrapper vector's length is set to the template
		 * length so that inner products work properly.
		 */

		time_series = gsl_vector_view_array((double *) gst_adapter_peek(element->adapter, (output_length + template_length(element) - 1) * sizeof(*time_series.vector.data)), template_length(element));

		/*
		 * Get buffers from the downstream peers, wrap both in GSL
		 * views.
		 */

		result = gst_pad_alloc_buffer(element->sumsquarespad, element->next_sample, output_length * sizeof(*orthogonal_snr_sum_squares.vector.data), GST_PAD_CAPS(element->sumsquarespad), &orthogonal_snr_sum_squares_buf);
		if(result != GST_FLOW_OK) {
			goto done;
		}

		orthogonal_snr_sum_squares = gsl_vector_view_array((double *) GST_BUFFER_DATA(orthogonal_snr_sum_squares_buf), output_length);

		result = gst_pad_alloc_buffer(element->srcpad, element->next_sample, num_templates(element) * output_length * sizeof(*orthogonal_snr.matrix.data), GST_PAD_CAPS(element->srcpad), &orthogonal_snr_buf);
		if(result != GST_FLOW_OK) {
			gst_buffer_unref(orthogonal_snr_sum_squares_buf);
			goto done;
		}

		orthogonal_snr = gsl_matrix_view_array((double *) GST_BUFFER_DATA(orthogonal_snr_buf), output_length, num_templates(element));

		/*
		 * Set the metadata.
		 */

		if(element->next_is_discontinuity) {
			GST_BUFFER_FLAG_SET(orthogonal_snr_sum_squares_buf, GST_BUFFER_FLAG_DISCONT);
			GST_BUFFER_FLAG_SET(orthogonal_snr_buf, GST_BUFFER_FLAG_DISCONT);
		}
		GST_BUFFER_OFFSET_END(orthogonal_snr_sum_squares_buf) = GST_BUFFER_OFFSET_END(orthogonal_snr_buf) = GST_BUFFER_OFFSET(orthogonal_snr_buf) + output_length;
		GST_BUFFER_TIMESTAMP(orthogonal_snr_sum_squares_buf) = GST_BUFFER_TIMESTAMP(orthogonal_snr_buf) = element->output_timestamp;
		GST_BUFFER_DURATION(orthogonal_snr_sum_squares_buf) = GST_BUFFER_DURATION(orthogonal_snr_buf) = (GstClockTime) output_length * GST_SECOND / element->sample_rate;

		/*
		 * Assemble the orthogonal SNR time series as the columns
		 * of a matrix (as the channels of a multi-channel audio
		 * stream).
		 */

		orthogonal_snr_sample = gsl_vector_alloc(orthogonal_snr.matrix.size2);
		if(!orthogonal_snr_sample) {
			GST_ERROR_OBJECT(element, "gsl_vector_new() failed");
			gst_buffer_unref(orthogonal_snr_sum_squares_buf);
			gst_buffer_unref(orthogonal_snr_buf);
			result = GST_FLOW_ERROR;
			goto done;
		}
		for(i = 0; i < output_length; i++) {
			/*
			 * The current row (time sample) in the output
			 * matrix.
			 */

			gsl_vector_view orthogonal_snr_row = gsl_matrix_row(&orthogonal_snr.matrix, i);

			/*
			 * Compute one vector of orthogonal SNR samples ---
			 * the projection of h(t) onto the template bank's
			 * orthonormal basis.
			 */

			gsl_blas_dgemv(CblasNoTrans, 1.0 / normalization_correction(element->sample_rate), element->U, &time_series.vector, 0.0, orthogonal_snr_sample);
			gsl_vector_memcpy(&orthogonal_snr_row.vector, orthogonal_snr_sample);

			/*
			 * From the projection of h(t) onto the bank's
			 * orthonormal basis, compute the square magnitude
			 * of the component of h(t) in the bank
			 */

			gsl_vector_mul(orthogonal_snr_sample, element->S);
			gsl_vector_set(&orthogonal_snr_sum_squares.vector, i, pow(gsl_blas_dnrm2(orthogonal_snr_sample) / num_templates(element), 2));

			/*
			 * Advance the time series pointer.
			 */

			time_series.vector.data++;
		}
		gsl_vector_free(orthogonal_snr_sample);

		/*
		 * Push the buffers downstream.
		 */

		result = gst_pad_push(element->sumsquarespad, orthogonal_snr_sum_squares_buf);
		if(result != GST_FLOW_OK) {
			gst_buffer_unref(orthogonal_snr_buf);
			goto done;
		}

		result = gst_pad_push(element->srcpad, orthogonal_snr_buf);
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Flush the data from the adapter that is no longer
		 * required, and advance the sample count.
		 */

		gst_adapter_flush(element->adapter, output_length * sizeof(*time_series.vector.data));
		element->next_is_discontinuity = FALSE;
		element->next_sample += output_length;
		element->output_timestamp += (GstClockTime) output_length * GST_SECOND / element->sample_rate;
	}

	/*
	 * Done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                            GObject Type Support
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
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	gst_object_unref(element->matrixpad);
	gst_object_unref(element->chifacspad);
	gst_object_unref(element->sumsquarespad);
	gst_object_unref(element->srcpad);
	g_object_unref(element->adapter);
	free(element->template_bank_filename);
	free(element->reference_psd_filename);

	svd_destroy(element);

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
		"Template Bank",
		"Filter",
		"A time-domain filter bank",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"matrix",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"chifacs",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sumofsquares",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
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

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = finalize;

	g_object_class_install_property(gobject_class, ARG_TEMPLATE_BANK, g_param_spec_string("template-bank", "XML Template Bank", "Name of LIGO Light Weight XML file containing inspiral template bank", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_REFERENCE_PSD, g_param_spec_string("reference-psd", "Reference PSD", "Name of file from which to read a reference PSD", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_T_START, g_param_spec_double("t-start", "Start time", "Start time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXDOUBLE, DEFAULT_T_START, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_T_END, g_param_spec_double("t-end", "End time", "End time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXDOUBLE, DEFAULT_T_END, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_T_TOTAL_DURATION, g_param_spec_double("t-total-duration", "Template total duration", "Total duration of the template (not just the piece being processed here) in seconds", 0, G_MAXDOUBLE, DEFAULT_T_END - DEFAULT_T_START, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SNR_LENGTH, g_param_spec_uint("snr-length", "SNR length", "Length, in samples, of the output SNR time series (0 = no limit)", 0, G_MAXUINT, DEFAULT_SNR_LENGTH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) matrix pad */
	element->matrixpad = gst_element_get_pad(GST_ELEMENT(element), "matrix");

	/* retrieve (and ref) chifacs pad */
	element->chifacspad = gst_element_get_pad(GST_ELEMENT(element), "chifacs");

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_pad(GST_ELEMENT(element), "src");

	/* retrieve (and ref) sum-of-squares pad */
	element->sumsquarespad = gst_element_get_pad(GST_ELEMENT(element), "sumofsquares");

	/* internal data */
	element->adapter = gst_adapter_new();

	element->reference_psd_filename = NULL;
	element->template_bank_filename = NULL;
	element->t_start = DEFAULT_T_START;
	element->t_end = DEFAULT_T_END;
	element->t_total_duration = 0;
	element->snr_length = DEFAULT_SNR_LENGTH;

	element->next_is_discontinuity = FALSE;
	element->next_sample = 0;
	element->output_timestamp = 0;

	element->U = NULL;
	element->S = NULL;
	element->V = NULL;
	element->chifacs = NULL;
}


/*
 * gstlal_templatebank_get_type().
 */


GType gstlal_templatebank_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTemplateBankClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALTemplateBank),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_templatebank", &info, 0);
	}

	return type;
}
