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
#define DEFAULT_T_END G_MAXUINT
#define DEFAULT_SNR_LENGTH 2048	/* samples */
#define TEMPLATE_DURATION 64	/* seconds */
#define CHIRPMASS_START 1.15	/* M_sun */
#define TEMPLATE_SAMPLE_RATE 2048	/* Hertz */
#define NUM_TEMPLATES 200
#define TOLERANCE 0.97


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


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
	 * TEMPLATE_DURATION (both are unsigned so can't be negative)
	 */

	if(element->t_start > TEMPLATE_DURATION)
		element->t_start = TEMPLATE_DURATION;
	if(element->t_end < element->t_start)
		element->t_end = element->t_start;
	else if(element->t_end > TEMPLATE_DURATION)
		element->t_end = TEMPLATE_DURATION;

	/*
	 * generate orthonormal template bank
	 */

	generate_bank_svd(&element->U, &element->S, &element->V, &element->chifacs, CHIRPMASS_START, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / sample_rate, NUM_TEMPLATES, element->t_start, element->t_end, TEMPLATE_DURATION, TOLERANCE, verbose);

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
	if(!gst_pad_set_caps(pad, caps)) {
		GST_ERROR("failure negotiating caps with mixer");
		gst_caps_unref(caps);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}
	gst_caps_unref(caps);

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


/*
 * Project some input data onto the template bank
 */


/*
 * ============================================================================
 *
 *                             GStreamer Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_T_START = 1,
	ARG_T_END,
	ARG_SNR_LENGTH
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	switch(id) {
	case ARG_T_START:
		element->t_start = g_value_get_uint(value);
		break;

	case ARG_T_END:
		element->t_end = g_value_get_uint(value);
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
	case ARG_T_START:
		g_value_set_uint(value, element->t_start);
		break;

	case ARG_T_END:
		g_value_set_uint(value, element->t_end);
		break;

	case ARG_SNR_LENGTH:
		g_value_set_uint(value, element->snr_length);
	}
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	element->sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	gboolean is_discontinuity = FALSE;
	GstFlowReturn result = GST_FLOW_OK;
	int output_length;
	int i;

	/*
	 * Now that we know the sample rate, construct orthogonal basis for
	 * the template bank if not already done.
	 */

	if(!element->U) {
		GstCaps *srccaps;
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
		 * the srcpad's caps properly.  gst_caps_make_writable()
		 * unref()s its argument so we have to ref() it first to
		 * keep it valid.
		 */

		success = gst_pad_set_caps(element->sumsquarespad, caps);
		if(success != TRUE) {
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		gst_caps_ref(caps);
		srccaps = gst_caps_make_writable(caps);
		gst_caps_set_simple(srccaps, "channels", G_TYPE_INT, element->U->size1, NULL);
		success = gst_pad_set_caps(element->srcpad, srccaps);
		gst_caps_unref(srccaps);
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

		is_discontinuity = TRUE;
		gst_adapter_clear(element->adapter);

		/*
		 * Pad the adapter with enough 0s to accomodate the
		 * template minus 1 sample, so that the first SNR sample
		 * generated is for when the first input sample intersects
		 * the start of the template.
		 */

		zeros = gst_buffer_try_new_and_alloc((element->U->size1 - 1) * sizeof(*element->U->data));
		if(!zeros) {
			result = GST_FLOW_ERROR;
			goto done;
		}
		memset(GST_BUFFER_DATA(zeros), 0, GST_BUFFER_SIZE(zeros));
		gst_adapter_push(element->adapter, zeros);

		/*
		 * The time of the start of the h(t) buffer from which the
		 * SNR buffer will be constructed is
		 * GST_BUFFER_TIMESTAMP(sinkbuf) - (element->U->size1 - 1)
		 * / sample_rate.  Relative to the time-of-coalescence ---
		 * the "time" of the template --- the first sample of the
		 * template vector is at -t_end + 1 * deltaT.  The "time"
		 * of an SNR sample is, therefore, the start of the h(t)
		 * buffer + t_end - 1*deltaT - (element->U->size1 -
		 * 1)*deltaT = buffer + t_end - element->U->size1*deltaT =
		 * buffer + t_start.
		 *
		 * FIXME:  that explanation sucks.  draw a picture.
		 */

		element->output_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf) + GST_SECOND * element->t_start;
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
		gsl_vector_view orthogonal_snr_sum_squares;

		/*
		 * Check for available data, clip to the required output
		 * length.  Wrap the data in a GSL vector view.
		 */

		output_length = (gst_adapter_available(element->adapter) / sizeof(*time_series.vector.data)) - element->U->size2;
		if(element->snr_length) {
			if(output_length < (int) element->snr_length)
				break;
			output_length = element->snr_length;
		} else if(output_length <= 0)
			break;

		time_series = gsl_vector_view_array((double *) gst_adapter_peek(element->adapter, (element->U->size2 + output_length - 1) * sizeof(*time_series.vector.data)), element->U->size2);

		/*
		 * Get buffers from the downstream peers, wrap both in GSL
		 * views.
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->next_sample, element->U->size1 * output_length * sizeof(*orthogonal_snr.matrix.data), GST_PAD_CAPS(element->srcpad), &orthogonal_snr_buf);
		if(result != GST_FLOW_OK)
			goto done;

		orthogonal_snr = gsl_matrix_view_array((double *) GST_BUFFER_DATA(orthogonal_snr_buf), output_length, element->U->size1);

		result = gst_pad_alloc_buffer(element->sumsquarespad, element->next_sample, output_length * sizeof(*orthogonal_snr_sum_squares.vector.data), GST_PAD_CAPS(element->sumsquarespad), &orthogonal_snr_sum_squares_buf);
		if(result != GST_FLOW_OK) {
			gst_buffer_unref(orthogonal_snr_buf);
			goto done;
		}

		orthogonal_snr_sum_squares = gsl_vector_view_array((double *) GST_BUFFER_DATA(orthogonal_snr_sum_squares_buf), output_length);

		/*
		 * Set the metadata.
		 */

		if(is_discontinuity) {
			GST_BUFFER_FLAG_SET(orthogonal_snr_sum_squares_buf, GST_BUFFER_FLAG_DISCONT);
			GST_BUFFER_FLAG_SET(orthogonal_snr_buf, GST_BUFFER_FLAG_DISCONT);
			is_discontinuity = FALSE;
		}
		GST_BUFFER_OFFSET_END(orthogonal_snr_sum_squares_buf) = GST_BUFFER_OFFSET_END(orthogonal_snr_buf) = GST_BUFFER_OFFSET(orthogonal_snr_buf) + output_length - 1;
		GST_BUFFER_TIMESTAMP(orthogonal_snr_sum_squares_buf) = GST_BUFFER_TIMESTAMP(orthogonal_snr_buf) = element->output_timestamp;
		GST_BUFFER_DURATION(orthogonal_snr_sum_squares_buf) = GST_BUFFER_DURATION(orthogonal_snr_buf) = output_length * GST_SECOND / element->sample_rate;

		/*
		 * Assemble the orthogonal SNR time series as the columns
		 * of a matrix (as the channels of a multi-channel audio
		 * stream).
		 */

		for(i = 0; i < output_length; i++) {
			/*
			 * The current row (time sample) in the output
			 * matrix.
			 */

			gsl_vector_view orthogonal_snr_sample = gsl_matrix_row(&orthogonal_snr.matrix, i);

			/*
			 * Compute one vector of orthogonal SNR samples ---
			 * the projection of h(t) onto the template bank's
			 * orthonormal basis.
			 */

			gsl_blas_dgemv(CblasNoTrans, 1.0, element->U, &time_series.vector, 0.0, &orthogonal_snr_sample.vector);

			/*
			 * From the projection of h(t) onto the bank's
			 * orthonormal basis, compute the square magnitude
			 * of the component of h(t) in the bank
			 */

			gsl_vector_set(&orthogonal_snr_sum_squares.vector, i, pow(gsl_blas_dnrm2(&orthogonal_snr_sample.vector), 2));

			/*
			 * Advance the time series pointer.
			 */

			time_series.vector.data++;
		}

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
		element->output_timestamp += (GstClockTime) output_length * GST_SECOND / element->sample_rate;
		element->next_sample += output_length;
	}

	/*
	 * Done
	 */

done:
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject *object)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	gst_object_unref(element->matrixpad);
	element->matrixpad = NULL;
	gst_object_unref(element->sumsquarespad);
	element->sumsquarespad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_object_unref(element->adapter);
	element->adapter = NULL;

	svd_destroy(element);

	G_OBJECT_CLASS(parent_class)->dispose(object);
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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
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
	gobject_class->dispose = dispose;

	g_object_class_install_property(gobject_class, ARG_T_START, g_param_spec_uint("t-start", "Start time", "Start time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXUINT, DEFAULT_T_START, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_T_END, g_param_spec_uint("t-end", "End time", "End time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXUINT, DEFAULT_T_END, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
	element->matrixpad = gst_element_get_static_pad(GST_ELEMENT(element), "matrix");

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* retrieve (and ref) sum-of-squares pad */
	element->sumsquarespad = gst_element_get_static_pad(GST_ELEMENT(element), "sumofsquares");

	/* internal data */
	element->adapter = gst_adapter_new();
	element->t_start = DEFAULT_T_START;
	element->t_end = DEFAULT_T_END;
	element->snr_length = DEFAULT_SNR_LENGTH;

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
